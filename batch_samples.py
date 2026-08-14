"""Generate one sample wav per voice preset, loading the model once.

    python batch_samples.py                      # every en- voice, CPU
    python batch_samples.py --prefix en-Carter   # a subset
    python batch_samples.py --num-threads 8 --ddpm-steps 3

Each sample says its own voice name, so the files are self-identifying when
played back. Running vibevoice.py once per voice would reload the 0.5B model
every time; this keeps one model in memory for the whole batch.
"""
from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging as hf_logging

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

ROOT = Path(__file__).resolve().parent
VOICES = ROOT / "demo" / "voices" / "streaming_model"
MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
SCRIPT = ("This is {name}. You are listening to Vibe Voice, a novel framework "
          "designed for generating expressive, long-form, multi-speaker "
          "conversational TTS.")


def log(msg: str):
    now = datetime.now()
    print(now.strftime("%Y-%m-%d ") + now.strftime("%I:%M:%S%p").lstrip("0") + f" - {msg}",
          flush=True)


def speaker_name(stem: str) -> str:
    """en-Carter_man -> Carter"""
    return stem.split("-", 1)[-1].rsplit("_", 1)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="en-", help="voice stems to include (default: en-)")
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "samples")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--ddpm-steps", type=int, default=5)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    args = p.parse_args()

    hf_logging.set_verbosity_error()
    torch.set_num_threads(args.num_threads)

    voices = sorted(p for p in VOICES.rglob("*.pt") if p.stem.startswith(args.prefix))
    if not voices:
        raise SystemExit(f"no voices matching '{args.prefix}' under {VOICES}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {MODEL_ID} on cpu (torch.float32, sdpa), {args.num_threads} threads")
    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_ID)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)
    log(f"{len(voices)} voices to render into {args.outdir}")

    batch_start = time.time()
    for i, voice in enumerate(voices, 1):
        out = args.outdir / f"{voice.stem}.wav"
        log(f"[{i}/{len(voices)}] {voice.stem}")
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            prefilled = torch.load(voice, map_location="cpu", weights_only=True)
        inputs = processor.process_input_with_cached_prompt(
            text=SCRIPT.format(name=speaker_name(voice.stem)), cached_prompt=prefilled,
            padding=True, return_tensors="pt", return_attention_mask=True,
        )
        t0 = time.time()
        outputs = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
            tokenizer=processor.tokenizer, generation_config={"do_sample": False},
            verbose=False, all_prefilled_outputs=copy.deepcopy(prefilled),
        )
        elapsed = time.time() - t0
        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            log(f"    no audio produced for {voice.stem}, skipping")
            continue
        processor.save_audio(outputs.speech_outputs[0], output_path=str(out))
        seconds = outputs.speech_outputs[0].shape[-1] / 24000
        log(f"    {out.name}: {seconds:.1f}s of audio in {elapsed:.0f}s "
            f"(RTF {elapsed / seconds:.2f}x)")

    log(f"Done in {(time.time() - batch_start) / 60:.1f} min -> {args.outdir}")


if __name__ == "__main__":
    main()
