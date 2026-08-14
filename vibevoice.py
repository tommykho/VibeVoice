"""Local text-to-speech on CPU or GPU, across both VibeVoice model stacks.

    python vibevoice.py --file speech.txt --voice Carter
    python vibevoice.py --cpu --num-threads 4 --file speech.txt
    python vibevoice.py --list-voices

Two stacks live behind --model, and the default picks between them by
hardware, so the same command suits a CUDA desktop and a CPU laptop:

  0.5b   streaming realtime model, shipped in this repository, works out of
         the box. Single speaker, voices are embedded .pt prefill states.
  1.5b   long-form multi-speaker model. Microsoft removed its inference code
         from this repository, so it needs the community fork installed; the
         error message says how.

--cpu forces float32 + sdpa on the CPU. Expect generation to be slower than
real time on laptop-class hardware; the run prints RTF so you can measure.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STREAMING_VOICES = ROOT / "demo" / "voices" / "streaming_model"
LONGFORM_VOICES = ROOT / "demo" / "voices"
OUTPUTS = ROOT / "outputs"

MODELS = {
    "0.5b": "microsoft/VibeVoice-Realtime-0.5B",
    "1.5b": "vibevoice/VibeVoice-1.5B",
    "7b": "vibevoice/VibeVoice-7B",
}
FORK_URL = "https://github.com/vibevoice-community/VibeVoice"


def stamp() -> str:
    """2026-08-13 9:08:11PM -- no leading zero on the hour."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d ") + now.strftime("%I:%M:%S%p").lstrip("0")


def log(msg: str):
    print(f"{stamp()} - {msg}")


def die(msg: str):
    print(f"{stamp()} - error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# -- device -----------------------------------------------------------

def resolve_model(requested: str) -> str:
    """auto -> 1.5b on a CUDA box with the fork installed, else 0.5b.

    Matches the usual split: the long-form model on a desktop GPU, the
    streaming model on a laptop without CUDA, from one unchanged command.
    """
    if requested != "auto":
        return requested
    import importlib.util
    import torch

    if not torch.cuda.is_available():
        return "0.5b"
    try:
        found = importlib.util.find_spec(
            "vibevoice.modular.modeling_vibevoice_inference"
        ) is not None
    except ModuleNotFoundError:
        found = False
    return "1.5b" if found else "0.5b"


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        die("torch is not installed. From the repository root: pip install -e .[streamingtts]")


def resolve_device(requested: str | None, force_cpu: bool) -> str:
    """-> cuda | mps | cpu. --cpu wins over everything, including --device."""
    import torch

    if force_cpu:
        return "cpu"
    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    device = "mps" if requested == "mpx" else requested
    if device == "cuda" and not torch.cuda.is_available():
        log("Warning: CUDA not available. Falling back to CPU.")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        log("Warning: MPS not available. Falling back to CPU.")
        return "cpu"
    return device


def load_options(device: str):
    """The dtype/attention triage every entry point in this repo repeats."""
    import torch

    if device == "cuda":
        return torch.bfloat16, "flash_attention_2", "cuda"
    if device == "mps":
        return torch.float32, "sdpa", None      # device_map="mps" is unsupported
    return torch.float32, "sdpa", "cpu"


# -- voices -----------------------------------------------------------

def voice_files(model_key: str) -> dict[str, Path]:
    if model_key == "0.5b":
        return {p.stem: p for p in sorted(STREAMING_VOICES.rglob("*.pt"))}
    return {p.stem: p for p in sorted(LONGFORM_VOICES.glob("*.wav"))}


def resolve_voice(name: str, model_key: str) -> Path:
    """Exact, then case-insensitive, then unique substring. Never guesses.

    This repository's own VoiceMapper falls back to an arbitrary voice when
    nothing matches, which yields a finished file in the wrong voice.
    """
    files = voice_files(model_key)
    if not files:
        where = STREAMING_VOICES if model_key == "0.5b" else LONGFORM_VOICES
        die(f"no voices in {where}")
    if name in files:
        return files[name]
    low = name.lower()
    for key in files:
        if key.lower() == low:
            return files[key]
    hits = [k for k in files if low in k.lower()]
    if len(hits) == 1:
        return files[hits[0]]
    if hits:
        die(f"voice '{name}' is ambiguous: {', '.join(hits)}")
    die(f"no voice matching '{name}'. Available: {', '.join(files)}")


# -- script parsing ---------------------------------------------------

SPEAKER_RE = re.compile(r"^Speaker\s+(\d+):\s*(.*)$", re.IGNORECASE)


def parse_script(text: str) -> list[tuple[int, str]]:
    """-> [(speaker_no, line)]. Plain prose collapses to one Speaker 1 segment."""
    out: list[tuple[int, str]] = []
    cur_no: int | None = None
    cur: list[str] = []
    for raw in text.replace("’", "'").strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            if cur_no is not None and cur:
                out.append((cur_no, " ".join(cur)))
            cur_no = int(m.group(1))
            cur = [m.group(2).strip()] if m.group(2).strip() else []
        elif cur_no is not None:
            cur.append(line)
        else:
            cur_no, cur = 1, [line]
    if cur_no is not None and cur:
        out.append((cur_no, " ".join(cur)))
    return out


# -- streaming 0.5b ---------------------------------------------------

def run_streaming(args, device: str):
    import copy
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    voice = resolve_voice(args.voice[0], "0.5b")
    script = " ".join(text for _, text in parse_script(args.file.read_text(encoding="utf-8")))
    if not script:
        die(f"{args.file} is empty")

    dtype, attn, device_map = load_options(device)
    log(f"Loading {args.model_id} on {device} ({dtype}, {attn})")
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_id)
    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            args.model_id, torch_dtype=dtype, device_map=device_map, attn_implementation=attn,
        )
    except Exception:
        if attn != "flash_attention_2":
            raise
        log("flash_attention_2 unavailable, retrying with sdpa (may lower audio quality).")
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            args.model_id, torch_dtype=dtype, device_map=device_map, attn_implementation="sdpa",
        )
    if device == "mps":
        model.to("mps")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    log(f"Voice: {voice.stem}")
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        prefilled = torch.load(voice, map_location=device, weights_only=True)

    inputs = processor.process_input_with_cached_prompt(
        text=script, cached_prompt=prefilled, padding=True,
        return_tensors="pt", return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    t0 = time.time()
    outputs = model.generate(
        **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer, generation_config={"do_sample": False},
        verbose=False, all_prefilled_outputs=copy.deepcopy(prefilled),
    )
    elapsed = time.time() - t0
    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        die("the model produced no audio")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=str(args.output))
    report(args, device, elapsed, outputs.speech_outputs[0].shape[-1] / 24000)


# -- long-form 1.5b / 7b ----------------------------------------------

def run_longform(args, device: str):
    import torch

    try:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
    except ImportError:
        die(
            f"the {args.model} model needs inference code that Microsoft removed from this\n"
            f"repository in 2025-09. Install the community fork, which retains it:\n"
            f"    pip install -e git+{FORK_URL}.git#egg=vibevoice\n"
            f"Or use --model 0.5b, which runs from this repository as-is."
        )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    segments = parse_script(args.file.read_text(encoding="utf-8"))
    if not segments:
        die(f"{args.file} is empty")
    nums = sorted({n for n, _ in segments})
    if len(nums) > 4:
        die(f"VibeVoice supports at most 4 speakers; the script has {len(nums)}")
    if nums[-1] > len(args.voice):
        die(f"script uses Speaker {nums[-1]} but only {len(args.voice)} voice(s) given")
    remap = {n: i + 1 for i, n in enumerate(nums)}
    paths = [resolve_voice(args.voice[n - 1], args.model) for n in nums]
    script = "\n".join(f"Speaker {remap[n]}: {t}" for n, t in segments)

    dtype, attn, device_map = load_options(device)
    log(f"Loading {args.model_id} on {device} ({dtype}, {attn})")
    processor = VibeVoiceProcessor.from_pretrained(args.model_id)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=device_map, attn_implementation=attn,
    )
    if device == "mps":
        model.to("mps")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    log(f"Voices: {', '.join(p.stem for p in paths)}")
    inputs = processor(
        text=[script], voice_samples=[[str(p) for p in paths]], padding=True,
        return_tensors="pt", return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    t0 = time.time()
    outputs = model.generate(
        **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer, generation_config={"do_sample": False},
        verbose=False, is_prefill=True,
    )
    elapsed = time.time() - t0
    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        die("the model produced no audio")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=str(args.output))
    report(args, device, elapsed, outputs.speech_outputs[0].shape[-1] / 24000)


def report(args, device: str, elapsed: float, seconds: float):
    import torch

    threads = f", {torch.get_num_threads()} threads" if device == "cpu" else ""
    log(f"Saved {args.output}")
    log(f"  {seconds:.1f}s of audio in {elapsed:.0f}s  "
        f"(RTF {elapsed / seconds:.2f}x on {device}{threads}, {args.ddpm_steps} ddpm steps)")


# -- selftest ---------------------------------------------------------

def selftest():
    """Assertions over the logic that would otherwise fail silently."""
    assert parse_script("Hello there.\nSecond line.") == [(1, "Hello there. Second line.")]
    assert parse_script("Speaker 1: Hi\nSpeaker 2: Bye\ntrailing\n") == [
        (1, "Hi"), (2, "Bye trailing")
    ]
    assert parse_script("speaker 3:  spaced  ") == [(3, "spaced")]
    assert parse_script("  \n\n") == []

    files = voice_files("0.5b")
    assert files, "the streaming presets should be present in a checkout"
    first = next(iter(files))
    assert resolve_voice(first, "0.5b") == files[first]
    assert resolve_voice(first.upper(), "0.5b") == files[first]
    # substring: "carter" -> en-Carter_man, and never a silent fallback
    assert resolve_voice("carter", "0.5b").stem == "en-Carter_man"
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            resolve_voice("definitely-not-a-voice", "0.5b")
        except SystemExit:
            pass
        else:
            raise AssertionError("an unmatched voice must raise, never fall back")

    assert set(MODELS) == {"0.5b", "1.5b", "7b"}
    print(f"selftest: all checks passed ({len(files)} streaming voices found)")


# -- cli --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Local text-to-speech with VibeVoice, on CPU or GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--file", type=Path, help=".txt/.md script")
    p.add_argument("--output", type=Path, help="output wav (default: outputs/<name>.wav)")
    p.add_argument("--voice", nargs="+", default=None,
                   help="one voice per speaker, in Speaker 1..N order")
    p.add_argument("--model", default="auto",
                   help=f"auto | {' | '.join(MODELS)} or a HuggingFace id. "
                        "auto picks 1.5b on a CUDA box with the community fork "
                        "installed, else 0.5b (default: auto)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU inference, overriding --device and auto-detection")
    p.add_argument("--num-threads", type=int, default=None,
                   help="CPU threads. On hybrid P/E-core laptops, matching the "
                        "P-core count is often faster than using every core.")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--ddpm-steps", type=int, default=5,
                   help="diffusion steps per acoustic frame; the main quality/speed dial")
    p.add_argument("--seed", type=int)
    p.add_argument("--list-voices", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="keep transformers' checkpoint and tokenizer warnings, which are "
                        "expected here: the unused encoder weights are never loaded")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if args.list_voices:
        files = voice_files(args.model if args.model in MODELS else "0.5b")
        if not files:
            print("no voices found")
        for name in files:
            print(f"  {name}")
        return
    if not args.file:
        p.error("--file is required")
    if not args.file.exists():
        die(f"{args.file} not found")
    if args.output is None:
        args.output = OUTPUTS / (args.file.stem + ".wav")

    require_torch()
    if not args.verbose:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    args.model = resolve_model(args.model)
    args.model_id = MODELS.get(args.model, args.model)
    if args.voice is None:
        args.voice = ["Carter"] if args.model == "0.5b" else ["custom"]
    device = resolve_device(args.device, args.cpu)
    if device == "cpu" and args.num_threads:
        import torch
        torch.set_num_threads(args.num_threads)
    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    if args.model == "0.5b":
        run_streaming(args, device)
    else:
        run_longform(args, device)


if __name__ == "__main__":
    main()
