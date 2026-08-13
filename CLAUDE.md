# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

Fork of `microsoft/VibeVoice` (origin: `tommykho/VibeVoice`). Research code, not a product: no test suite, no linter config, no CI. The two files under `vllm_plugin/tests/` are manual integration scripts that require an already-running vLLM server — they are not unit tests.

Install is editable-only:

```bash
pip install -e .                 # base (ASR + streaming TTS)
pip install -e .[streamingtts]   # pins transformers==4.51.3, required for the 0.5B streaming model
```

## The three model families and what code actually exists for each

`vibevoice/` looks like one package but holds three unrelated model stacks that share only the acoustic tokenizer and diffusion head. Knowing which stack a file belongs to is the single most useful orientation fact.

| Stack | Config | Model | Processor | Runnable entry point |
|---|---|---|---|---|
| **Streaming TTS 0.5B** | `configuration_vibevoice_streaming.py` | `modeling_vibevoice_streaming.py` + `modeling_vibevoice_streaming_inference.py` | `vibevoice_streaming_processor.py` | `demo/realtime_model_inference_from_file.py`, `demo/vibevoice_realtime_demo.py` → `demo/web/app.py` |
| **ASR 7B** | `configuration_vibevoice.py` (`VibeVoiceASRConfig`) | `modeling_vibevoice_asr.py` | `vibevoice_asr_processor.py` | `demo/vibevoice_asr_inference_from_file.py`, `demo/vibevoice_asr_gradio_demo.py`, `vllm_plugin/` |
| **Long-form TTS 1.5B** | `configuration_vibevoice.py` (`VibeVoiceConfig`) | `modeling_vibevoice.py` | `vibevoice_processor.py` | **none in-repo — needs the community fork** |

`vibevoice.py` at the repository root sits above both TTS stacks: `--model auto` picks 1.5B on a CUDA box where `vibevoice.modular.modeling_vibevoice_inference` imports, and the streaming 0.5B model otherwise. It imports torch lazily, so `--selftest` and `--list-voices` work without it.

### The 1.5B TTS path is deliberately incomplete

Upstream removed the VibeVoice-TTS inference code in 2025-09 ("used in ways inconsistent with the stated intent" — see README News). What survives is the *training-shaped* half:

- `VibeVoiceForConditionalGeneration` (`modeling_vibevoice.py`) has `forward()` and `forward_speech_features()` but **no `generate()`**. The `VibeVoiceForConditionalGenerationInference` subclass that carried generation, CFG, and the DDPM sampling loop is gone.
- `VibeVoiceProcessor` (`vibevoice_processor.py`) is intact — multi-speaker script parsing, `[1]:`/`[2]:` turn parsing, wav voice-prompt encoding, `save_audio()`.
- `demo/voices/*.wav` (Alice, Frank, …) and the 1.5B gradio/file demos are gone.
- Neither the 1.5B model nor processor is exported from `vibevoice/__init__.py` — only the streaming stack is.
- This history is **not recoverable from git**: the fork carries only 54 commits, all post-removal.

Any task touching 1.5B TTS inference means writing or porting a generation loop, not wiring up existing code. Do not assume the missing pieces are somewhere in the tree. The MIT-licensed community fork at `https://github.com/vibevoice-community/VibeVoice` retains the removed `modeling_vibevoice_inference.py`; weights stayed on HuggingFace under `vibevoice/VibeVoice-1.5B` and `vibevoice/VibeVoice-7B`.

## Voice presets are prefilled caches, not audio

For the streaming 0.5B model, `demo/voices/streaming_model/*.pt` are **precomputed prefill states**, not wav files — a deliberate deepfake mitigation ("voice prompts are provided in an embedded format", `docs/vibevoice-realtime-0.5b.md`). Each `.pt` deserializes to a dict containing `tts_lm` (a `BaseModelOutputWithPast`) and a `DynamicCache`, and is loaded under a safe-globals allowlist:

```python
with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
    prefilled = torch.load(path, map_location=device, weights_only=True)
```

Keep `weights_only=True` — commit 303b283 fixed CWE-502 here; do not regress it. The cache is consumed twice per generation, so callers pass `copy.deepcopy(all_prefilled_outputs)` into `model.generate()` while the un-copied dict feeds `processor.process_input_with_cached_prompt()`.

There is no in-repo path to *create* a new `.pt` from a wav — the encoder side was not shipped. Extra voices come from `bash demo/download_experimental_voices.sh`, which unpacks into `demo/voices/streaming_model/experimental_voices/`. `VoiceMapper` (in `realtime_model_inference_from_file.py`) globs `**/*.pt` recursively, lowercases the stem as the key, and does exact-then-substring matching, raising on ambiguous substring hits.

## Device / dtype conventions

Every entry point repeats the same triage, and new code should match it rather than invent its own:

- `cuda` → `torch.bfloat16` + `flash_attention_2`, `device_map="cuda"`
- `mps` → `torch.float32` + `sdpa`, `device_map=None` then `.to("mps")` (device_map="mps" is unsupported)
- `cpu` / `xpu` → `torch.float32` + `sdpa`, `device_map="cpu"`

`flash_attention_2` failures are caught and retried with `sdpa` (see `realtime_model_inference_from_file.py:201`). `demo/vibevoice_asr_gradio_demo.py:917` (`_detect_device_and_attn`) is the most complete version of this logic. Note `demo/vibevoice_realtime_demo.py` accepts a `mpx` typo and normalizes it to `mps`; `demo/web/app.py` does the same.

After loading any TTS model: `model.eval()` then `model.set_ddpm_inference_steps(num_steps=5)`. This count is the main quality/speed dial — the diffusion head runs that many steps per 7.5 Hz acoustic frame.

## Running things

```bash
# Unified CLI (model auto-selected by hardware); --selftest is the only test in the repo
python vibevoice.py --file speech.txt --voice Carter
python vibevoice.py --cpu --num-threads 4 --file speech.txt
python vibevoice.py --selftest

# Streaming TTS from a text file
python demo/realtime_model_inference_from_file.py \
  --model_path microsoft/VibeVoice-Realtime-0.5B \
  --txt_path demo/text_examples/1p_vibevoice.txt \
  --speaker_name Carter          # writes ./outputs/<txt-stem>_generated.wav

# Streaming TTS websocket demo (uvicorn → demo/web/app.py, reads MODEL_PATH/MODEL_DEVICE env)
python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B --port 3000

# ASR from a file
python demo/vibevoice_asr_inference_from_file.py --model_path microsoft/VibeVoice-ASR --audio_path <file>

# ASR via vLLM server (see docs/setup_gradio_demo.md for the full docker recipe)
python vllm_plugin/scripts/start_server.py --port 6001
```

Output audio is 24 kHz mono. RTF is printed by the file-inference scripts.

## Conventions from CONTRIBUTING.md

Upstream reviews line-by-line and explicitly rejects: over-engineering / added abstraction layers, formatting-only changes, and non-English comments or commit messages. It also calls out AI-generated code as a rejection risk unless heavily pruned. Keep diffs minimal and justify each added line; prefer editing an existing entry point over introducing a new module.
