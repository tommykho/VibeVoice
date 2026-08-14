<div align="center">

## 🎙️ VibeVoice: Open-Source Frontier Voice AI
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=githubpages)](https://microsoft.github.io/VibeVoice)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
[![TTS Report](https://img.shields.io/badge/TTS-Report-red?logo=arxiv)](https://openreview.net/pdf?id=FihSkzyxdv)
[![ASR Report](https://img.shields.io/badge/ASR-Report-yellow?logo=arxiv)](https://arxiv.org/pdf/2601.18184)
[![Colab](https://img.shields.io/badge/StreamingTTS-Colab-green?logo=googlecolab)](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb)
[![ASR Playground](https://img.shields.io/badge/ASR-Playground-6F42C1?logo=gradio)](https://aka.ms/vibevoice-asr)

[![microsoft%2FVibeVoice | Trendshift](https://trendshift.io/api/badge/repositories/15465)](https://trendshift.io/repositories/15465)

</div>


<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Figures/VibeVoice_logo_white.png">
  <img src="Figures/VibeVoice_logo.png" alt="VibeVoice Logo" width="300">
</picture>
</div>

<div align="left">

<h3>📰 News</h3>

<strong>2026-07-23: ⚡ We released <a href="https://github.com/microsoft/VibeASR.cpp">VibeVoice-ASR-BitNet</a>, an edge CPU inference engine for VibeVoice-ASR. Through heterogeneous quantization (I8_S + I2_S), the model is compressed from 4.62 GB to 1.58 GB with real-time inference (RTF < 1) on 3+ CPU threads — no GPU required. [<a href="https://github.com/microsoft/VibeASR.cpp">Code</a>] [<a href="https://huggingface.co/microsoft/VibeVoice-ASR-BitNet">Models</a>] [<a href="https://arxiv.org/abs/2607.21075">Report</a>]</strong>

<strong>2026-03-12: 🚀 VibeVoice-ASR is now integrated into <a href="https://labs.ai.azure.com/innovations/vibevoice-asr/">Azure AI Foundry Labs</a>! You can now explore and test our unified speech-to-text capabilities directly through Microsoft Foundry.</strong>

<strong>2026-03-06: 🚀 VibeVoice ASR is now part of a <a href="https://huggingface.co/microsoft/VibeVoice-ASR-HF">Transformers release</a>! You can now use our speech recognition model directly through the Hugging Face Transformers library for seamless integration into your projects.</strong>

<strong>2026-01-21:</strong> 📣 We open-sourced <a href="docs/vibevoice-asr.md"><strong>VibeVoice-ASR</strong></a>, a unified speech-to-text model designed to handle 60-minute long-form audio in a single pass, generating structured transcriptions containing Who (Speaker), When (Timestamps), and What (Content), with support for User-Customized Context. Try it in [Playground](https://aka.ms/vibevoice-asr).
- ⭐️ VibeVoice-ASR is natively multilingual, supporting over 50 languages — check the [supported languages](docs/vibevoice-asr.md#language-distribution) for details.
- 🔥 The VibeVoice-ASR [finetuning code](finetuning-asr/README.md) is now available!
- ⚡️ **vLLM inference** is now supported for faster inference; see [vllm-asr](docs/vibevoice-vllm-asr.md) for more details.
- 📑 [VibeVoice-ASR Technique Report](https://arxiv.org/pdf/2601.18184) is available.

2025-12-16: 📣 We added experimental speakers to <a href="docs/vibevoice-realtime-0.5b.md"><strong>VibeVoice‑Realtime‑0.5B</strong></a> for exploration, including multilingual voices in nine languages (DE, FR, IT, JP, KR, NL, PL, PT, ES) and 11 distinct English style voices. [Try it](docs/vibevoice-realtime-0.5b.md#optional-more-experimental-voices). More speaker types will be added over time.

2025-12-03: 📣 We open-sourced <a href="docs/vibevoice-realtime-0.5b.md"><strong>VibeVoice‑Realtime‑0.5B</strong></a>, a real‑time text‑to‑speech model that supports streaming text input and robust long-form speech generation. Try it on [Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb).


2025-09-05: VibeVoice is an open-source research framework intended to advance collaboration in the speech synthesis community. After release, we discovered instances where the tool was used in ways inconsistent with the stated intent. Since responsible use of AI is one of Microsoft’s guiding principles, we have removed the VibeVoice-TTS code from this repository.


2025-08-25: 📣 We open-sourced <a href="docs/vibevoice-tts.md"><strong>VibeVoice-TTS</strong></a>, a long-form multi-speaker text-to-speech model that can synthesize speech up to 90 minutes long with up to 4 distinct speakers. — accepted as an [Oral](https://openreview.net/forum?id=FihSkzyxdv) at ICLR 2026! 🔥

</div>

## Installation

Python 3.10 or later, plus `ffmpeg` on PATH. The streaming 0.5B model pins `transformers==4.51.3`.

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -r requirements.txt   # dependencies only
pip install -e .                  # puts the vibevoice package on the path
```

`pip install -e .[streamingtts]` installs both in one step; `requirements.txt` exists for the common case of installing dependencies without the editable package.

### Choosing a torch build

A bare `torch` does not mean the same thing everywhere: on Linux pip installs a CUDA build that carries several GB of NVIDIA libraries, on Windows a CPU-only one. Install torch explicitly *before* the rest, and the dependency is already satisfied when `requirements.txt` is processed.

```bash
# NVIDIA GPU — pick the index matching your driver; cu128 covers Blackwell
pip install "torch<2.10" --index-url https://download.pytorch.org/whl/cu128

# CPU only — avoids downloading CUDA libraries that will never be used
pip install "torch<2.10" --index-url https://download.pytorch.org/whl/cpu
```

Carry the `<2.10` pin on that command as well. Without it pip installs the newest torch, `requirements.txt` then downgrades it, and any companion package resolved against the newer version is left stranded — `torchvision` pins torch exactly and reports the mismatch as a dependency conflict. Neither `torchvision` nor `torchaudio` is used by this repository, so neither should be installed; if one is already present and conflicting, `pip uninstall torchvision` resolves it.

`flash_attention_2` is optional; every entry point falls back to `sdpa` when it is unavailable.

### Windows

Use Python 3.11: `transformers==4.51.3` has no wheels for 3.12+ on Windows. `git` and `ffmpeg` must be on PATH. Set `HF_HUB_DISABLE_SYMLINKS=1` before downloading weights, or the HuggingFace cache fails with `WinError 1314` unless Developer Mode is enabled.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_DISABLE_SYMLINKS = '1'
pip install "torch<2.10" --index-url https://download.pytorch.org/whl/cu128   # NVIDIA only
pip install -r requirements.txt
pip install -e .
```

Keep the venv on a local disk rather than a network share — torch is several GB of DLLs read on every import.

### CPU only

Install the CPU torch build above, then `pip install -r requirements.txt`. Use the 0.5B streaming model and see [CPU inference](#cpu-inference).

## Overview

VibeVoice is a **family of open-source frontier voice AI models** that includes both Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) models. 

A core innovation of VibeVoice is its use of continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of **7.5 Hz**. These tokenizers efficiently preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice employs a [next-token diffusion](https://arxiv.org/abs/2412.08635) framework, leveraging a Large Language Model (LLM) to understand textual context and dialogue flow, and a diffusion head to generate high-fidelity acoustic details.

For more information, demos, and examples, please visit our [Project Page](https://microsoft.github.io/VibeVoice).


<div align="center">

| Model |   Weight | Quick Try |
|-------|--------------|---------|
| VibeVoice-ASR-7B | [HF Link](https://huggingface.co/microsoft/VibeVoice-ASR) |  [Playground](https://aka.ms/vibevoice-asr) |
| VibeVoice-ASR-BitNet (CPU) | [HF Link](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) | [VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) |
| VibeVoice-TTS-1.5B | [HF Link](https://huggingface.co/microsoft/VibeVoice-1.5B) | Disabled |
| VibeVoice-Realtime-0.5B | [HF Link](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | [Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb) |

</div>

## Models


### 1. 📖 [VibeVoice-ASR](docs/vibevoice-asr.md) - Long-form Speech Recognition

**VibeVoice-ASR** is a unified speech-to-text model designed to handle **60-minute long-form audio** in a single pass, generating structured transcriptions containing **Who (Speaker), When (Timestamps), and What (Content)**, with support for **Customized Hotwords**.

- **🕒 60-minute Single-Pass Processing**:
  Unlike conventional ASR models that slice audio into short chunks (often losing global context), VibeVoice ASR accepts up to **60 minutes** of continuous audio input within 64K token length. This ensures consistent speaker tracking and semantic coherence across the entire hour.

- **👤 Customized Hotwords**:
  Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process, significantly improving accuracy on domain-specific content.

- **📝 Rich Transcription (Who, When, What)**:
  The model jointly performs ASR, diarization, and timestamping, producing a structured output that indicates *who* said *what* and *when*.

[📖 Documentation](docs/vibevoice-asr.md) | [🤗 Hugging Face](https://huggingface.co/microsoft/VibeVoice-ASR) | [🎮 Playground](https://aka.ms/vibevoice-asr) | [🛠️ Finetuning](finetuning-asr/README.md) |  [📊 Paper](docs/VibeVoice-ASR-Report.pdf)


<p align="center">
  <img src="Figures/DER.jpg" alt="DER" width="50%"><br>
  <img src="Figures/cpWER.jpg" alt="cpWER" width="50%"><br>
  <img src="Figures/tcpWER.jpg" alt="tcpWER" width="50%">
</p>


<div align="center" id="vibevoice-asr">

https://github.com/user-attachments/assets/acde5602-dc17-4314-9e3b-c630bc84aefa

</div>
<br>

### 2. 🎙️ [VibeVoice-TTS](docs/vibevoice-tts.md) - Long-form Multi-speaker TTS

**Best for**: Long-form conversational audio, podcasts, multi-speaker dialogues

- **⏱️ 90-minute Long-form Generation**:
  Synthesizes conversational/single-speaker speech up to **90 minutes** in a single pass, maintaining speaker consistency and semantic coherence throughout.

- **👥 Multi-speaker Support**:
  Supports up to **4 distinct speakers** in a single conversation, with natural turn-taking and speaker consistency across long dialogues.

- **🎭 Expressive Speech**:
  Generates expressive, natural-sounding speech that captures conversational dynamics and emotional nuances.

- **🌐 Multi-lingual Support**:
  Supports English, Chinese and other languages.


[📖 Documentation](docs/vibevoice-tts.md) | [🤗 Hugging Face](https://huggingface.co/microsoft/VibeVoice-1.5B)  |  [📊 Paper](https://arxiv.org/pdf/2508.19205)


<div align="center">
  <img src="Figures/VibeVoice-TTS-results.jpg" alt="VibeVoice Results" width="80%">
</div>


**English**
<div align="center">

https://github.com/user-attachments/assets/0967027c-141e-4909-bec8-091558b1b784

</div>


**Chinese**
<div align="center">

https://github.com/user-attachments/assets/322280b7-3093-4c67-86e3-10be4746c88f

</div>

**Cross-Lingual**
<div align="center">

https://github.com/user-attachments/assets/838d8ad9-a201-4dde-bb45-8cd3f59ce722

</div>

**Spontaneous Singing**
<div align="center">

https://github.com/user-attachments/assets/6f27a8a5-0c60-4f57-87f3-7dea2e11c730

</div>


**Long Conversation with 4 people**
<div align="center">

https://github.com/user-attachments/assets/a357c4b6-9768-495c-a576-1618f6275727

</div>





<br>

### 3. ⚡ [VibeVoice-Streaming](docs/vibevoice-realtime-0.5b.md) - Real-time Streaming TTS

VibeVoice-Realtime is a **lightweight real‑time** text-to-speech model supporting **streaming text input** and **robust long-form speech generation**.

- Parameter size: 0.5B (deployment-friendly)
- Real-time TTS (~300 milliseconds first audible latency)
- Streaming text input
- Robust long-form speech generation (~10 minutes)

[📖 Documentation](docs/vibevoice-realtime-0.5b.md) | [🤗 Hugging Face](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | [🚀 Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb)

#### Built-in voices

25 voice presets ship in `demo/voices/streaming_model/` as precomputed prefill states (`.pt`), not audio. Pass any of them to `--speaker_name`; matching is case-insensitive and accepts a unique substring, so `--speaker_name Carter` resolves `en-Carter_man`.

| Language | Presets |
|---|---|
| English | `en-Carter_man`, `en-Davis_man`, `en-Frank_man`, `en-Mike_man`, `en-Emma_woman`, `en-Grace_woman` |
| English (Indian) | `in-Samuel_man` |
| German | `de-Spk0_man`, `de-Spk1_woman` |
| French | `fr-Spk0_man`, `fr-Spk1_woman` |
| Italian | `it-Spk1_man`, `it-Spk0_woman` |
| Japanese | `jp-Spk0_man`, `jp-Spk1_woman` |
| Korean | `kr-Spk1_man`, `kr-Spk0_woman` |
| Dutch | `nl-Spk0_man`, `nl-Spk1_woman` |
| Polish | `pl-Spk0_man`, `pl-Spk1_woman` |
| Portuguese | `pt-Spk1_man`, `pt-Spk0_woman` |
| Spanish | `sp-Spk1_man`, `sp-Spk0_woman` |

#### Installing the experimental voices

A further set of experimental speakers is distributed separately — 11 English styles plus additional voices in German, French, Japanese, Korean, Polish, Portuguese and Spanish:

```bash
bash demo/download_experimental_voices.sh
python vibevoice.py --list-voices          # confirm the new names appear
```

The script fetches nine archives and unpacks them into `demo/voices/streaming_model/experimental_voices/`. Both the demo scripts and `vibevoice.py` scan that directory recursively, so the new voices need no further configuration and are used exactly like the built-in ones.

On Windows, run it from Git Bash (installed with Git for Windows); the script uses `wget` or `curl`, whichever is present. Re-running is safe — each archive is deleted after extraction.

Non-English voices are exploratory: the model is trained for English, and other languages may produce unpredictable results.

Custom voices cannot be created from this repository: the presets are embedded prefill states and the encoder that produces them was not released, a deliberate deepfake mitigation.

#### CPU inference

Pass `--cpu` to run without a GPU (float32 + sdpa). `--num_threads` caps the thread count, which is usually worth tuning on hybrid P/E-core laptops:

```bash
python demo/realtime_model_inference_from_file.py --cpu --num_threads 4 \
  --txt_path demo/text_examples/1p_vibevoice.txt --speaker_name Carter
```

The model needs roughly 2 GB of weights plus activations at float32, so 8 GB of RAM is comfortable. Expect generation to be slower than real time on laptop-class CPUs; the run prints RTF so you can measure your own. Lowering the DDPM step count is the main speed dial.

## `vibevoice.py` — one CLI for both TTS stacks

`vibevoice.py` at the repository root wraps both text-to-speech models behind a single command. `--model` defaults to `auto`, which selects the long-form 1.5B model on a CUDA machine that has the community fork installed and the streaming 0.5B model everywhere else — so the same command line suits a desktop GPU and a CPU laptop.

```bash
python vibevoice.py --file speech.txt --voice Carter     # auto-selects the model
python vibevoice.py --cpu --num-threads 4 --file speech.txt
python vibevoice.py --model 1.5b --file dialogue.txt --voice alice frank
python vibevoice.py --list-voices
python vibevoice.py --selftest
```

Scripts are plain text; `Speaker 1:` / `Speaker 2:` prefixes split turns for the multi-speaker 1.5B model, and prose without prefixes becomes a single speaker. Voice names resolve exactly, then case-insensitively, then by unique substring, and an unmatched name is an error rather than a silent fallback to some other voice.

Selecting `--model 1.5b` or `7b` requires the inference code Microsoft removed from this repository in September 2025; `vibevoice.py` reports how to install the community fork that retains it. `--model 0.5b` runs from this repository with no extra setup.


<div align="center" id="generated-example-audio-vibevoice-realtime">

https://github.com/user-attachments/assets/0901d274-f6ae-46ef-a0fd-3c4fba4f76dc

</div>

<br>

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.



## ⚠️ Risks and Limitations


While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 1.5b in this release).
Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.


We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only. Please use responsibly.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=Microsoft/vibevoice&type=date&legend=top-left)
