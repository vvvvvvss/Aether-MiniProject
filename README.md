# 🕵️ ForensicFace AI

**Sketch-to-Realistic Face Reconstruction for Forensic Investigations**

A mini project by the Department of Computer Science and Engineering, UVCE (2025–2026), under the guidance of **Dr. Pushpa C N**, Associate Professor.

ForensicFace AI converts rough forensic sketches into photorealistic face composites using generative AI, guided by a witness's verbal description. It ships with a simple web dashboard so non-technical investigators can use it without writing any code.

> ⚠️ **For investigative assistance only.** Outputs are AI-generated synthetic faces. They are not legal evidence and must be verified by a qualified forensic expert.

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [System Architecture](#-system-architecture)
5. [Pipeline](#-pipeline)
6. [Repository Structure](#-repository-structure)
7. [Getting Started](#-getting-started)
8. [Usage](#-usage)
9. [Results](#-results)
10. [Extension Notebook](#-extension-notebook-pkl-quantization-distillation)
11. [Challenges Faced](#-challenges-faced)
12. [Future Scope](#-future-scope)
13. [Ethical Considerations](#-ethical-considerations)
14. [Team](#-team)

---

## ❓ Problem Statement

- Traditional forensic sketching needs skilled artists, who are not available everywhere.
- Human memory is fallible, so sketches often need multiple revisions.
- Existing tools are not accessible to non-technical investigators.

**Our solution:** turn a rough sketch plus a verbal description into photorealistic composites through a simple web interface.

---

## ✨ Features

- Sketch-to-face generation that preserves face shape, jaw, nose bridge and eye placement
- Verbal description control (age, gender, skin tone, beard, scars and other descriptors)
- Four sketch preprocessing modes: **Auto, Scribble, Canny, Raw**
- 1–8 composite variations per run
- Adjustable guidance scale, ControlNet strength, inference steps and seed
- Fixed-seed reproducibility for case documentation
- Gradio dashboard with edge-map preview, output gallery and status bar
- Legal disclaimer displayed on every launch

---

## 🧰 Tech Stack

| Component | Role |
|-----------|------|
| **Stable Diffusion v1.5** | Core generative model (latent diffusion: VAE + U-Net) |
| **ControlNet (Scribble)** | Enforces the sketch's structure on the generated face |
| **CLIP Text Encoder** | Encodes descriptions into 768-dim embeddings that steer generation |
| **UniPC Scheduler** | Fast sampler, good quality in 20–30 steps |
| **OpenCV** | Sketch to edge-map preprocessing |
| **Gradio** | Web interface |
| **PyTorch / Diffusers** | Model loading and inference |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│ Presentation Layer                          │
│ Gradio UI: upload, demographics, gallery    │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ Processing Layer                            │
│ OpenCV preprocessing → CLIP prompt →        │
│ ControlNet conditioning                     │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ Model Layer                                 │
│ Stable Diffusion v1.5 · ControlNet Scribble │
│ UniPC Scheduler · CLIP · VAE                │
└─────────────────────────────────────────────┘
```

---

## 🔄 Pipeline

1. **Upload sketch**: PNG/JPEG, auto-resized to 512×512
2. **Preprocess**: OpenCV converts the sketch to an edge map
3. **Build prompt**: description and demographics are encoded by CLIP
4. **ControlNet**: edge map becomes spatial conditioning signals
5. **U-Net denoise**: 30 steps guided by CLIP and ControlNet
6. **VAE decode**: latent tensor becomes an RGB image
7. **Output gallery**: variations are displayed and saved to `forensic_outputs/`

---

## 📁 Repository Structure

```
ForensicFaceAI/
├── sketch_to_real_face.ipynb          # Main notebook: pipeline + Gradio UI
├── forensicface_pkl_quantization.ipynb# Extension: PKL classifiers, FP16/INT8, distillation
├── forensic_outputs/                  # Generated composites (created at runtime)
├── pkl_models/                        # Saved .pkl classifiers (extension notebook)
├── ForensicFaceAI_Documentation.docx  # Technical documentation
└── README.md
```

---

## 🚀 Getting Started

### Recommended environment

| Item | Requirement |
|------|-------------|
| Platform | Google Colab (free T4 GPU is sufficient) |
| Python | 3.9 – 3.11 |
| GPU | NVIDIA with 6 GB+ VRAM (CPU works but is very slow) |
| Disk | ~10 GB free for model cache |
| First-run download | ~5.4 GB (Stable Diffusion + ControlNet weights) |

### Run on Google Colab (easiest)

1. Upload `sketch_to_real_face.ipynb` to Colab.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order.
4. Open the public Gradio link printed by the last UI cell.

### Run locally

```bash
python -m venv forensicface-env
source forensicface-env/bin/activate        # Windows: forensicface-env\Scripts\activate

pip install diffusers transformers accelerate controlnet-aux \
    torch torchvision opencv-python-headless Pillow gradio \
    xformers safetensors

jupyter notebook sketch_to_real_face.ipynb
```

> `xformers` is CUDA-only. Skip it on CPU or Apple Silicon.

---

## 🖥️ Usage

1. **Upload** a forensic sketch (black lines on a white background work best).
2. **Describe** the suspect, e.g. `square jaw, thin beard, close-set eyes`.
3. **Select** gender, age range and skin tone.
4. Pick a **preprocessing mode** (`auto` suits most hand-drawn sketches).
5. Optionally open **Advanced Settings**:

| Setting | Default | Notes |
|---------|---------|-------|
| Variations | 4 | 1–8 |
| Guidance scale | 7.5 | Higher follows the text more closely |
| ControlNet strength | 0.85 | Higher follows the sketch more closely |
| Inference steps | 30 | 20 for speed, 40–50 for quality |
| Seed | -1 | Fixed seed gives identical output |

6. Click **Generate Composite Faces**. Images are shown in the gallery and saved to `forensic_outputs/`.

**Tuning tips**

- Output drifts from the sketch → raise ControlNet strength.
- Output looks rigid or artifacted → lower ControlNet strength (below ~0.9).
- Rough, messy sketch → use `scribble`. Clean line art → use `canny`.

---

## 📊 Results

| Aspect | Outcome |
|--------|---------|
| Structural fidelity | Face shape, jaw, nose bridge and eye placement preserved from the sketch |
| Text alignment | Skin tone, age, gender and descriptors (beard, scars) applied correctly |
| FID score | **28–32** on local GPU (plateaued) → **14–18** on Colab T4 |
| Speed | 4 variations in under 2 minutes on a T4 |
| Reproducibility | Fixed seeds produce identical outputs |

> 📸 *Add screenshots here: dashboard, preprocessed edge map, generated composites.*
>
> `![Dashboard](assets/dashboard.png)`
> `![Results](assets/results.png)`

---

## 🧪 Extension Notebook: PKL, Quantization, Distillation

`forensicface_pkl_quantization.ipynb` explores how `.pkl` models and model compression can complement the main pipeline:

- **PKL attribute classifiers**: predict age, gender and skin tone from sketch features to pre-fill the UI.
- **PKL quality ranker**: sorts generated composites best-first.
- **Knowledge distillation**: a Teacher MLP distilled into a compact scikit-learn student saved as `.pkl`.
- **Quantization**: FP16 for GPUs, dynamic INT8 for CPU, ONNX export path for edge devices.

> **Note:** the classifiers in this notebook are trained on a **synthetic placeholder dataset** to demonstrate the workflow. To use them for real predictions, retrain on a labeled forensic sketch dataset using `extract_sketch_features()`.

---

## 🧗 Challenges Faced

- **Compute limits:** Stable Diffusion + ControlNet is slow on CPU and low-VRAM GPUs, so we moved to Google Colab (T4).
- **Dependency conflicts:** `diffusers`, `torch`, `xformers` and `controlnet-aux` needed carefully matched versions.
- **Early output quality:** weak sketch adherence was fixed by tuning ControlNet strength, preprocessing mode and prompt design.

---

## 🔭 Future Scope

- Real-time generation with live investigator feedback
- Age progression / regression for cold cases
- Multi-sketch fusion into a single consensus composite
- Mobile app for field investigators
- Fine-tuning Stable Diffusion on curated forensic sketch-to-photo pairs

---

## ⚖️ Ethical Considerations

- All outputs are **synthetic**; the system does not query any biometric or criminal database.
- Composites are **investigative aids only** and need expert verification before use.
- Victim and case data must be handled under applicable data protection laws.
- Generated images should not be published without proper legal authorization.
- AI face generation can reflect biases from its training data (LAION-5B); treat outputs as one input among many, never as identification.

---

## 👥 Team

**Guide:** Dr. Pushpa C N, Associate Professor, Dept. of CSE, UVCE

**Team members** (6th Sem, B.Tech CSE):

- Maheshwari K (U25UV23T006029)
- Unnathi R B (U25UV23T006056)
- Varsha Rao (U25UV23T006059)
- Varsha Shubhashri M (U25UV23T006060)

---

## 🙏 Acknowledgements

- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [ControlNet](https://github.com/lllyasviel/ControlNet) by Lvmin Zhang
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Gradio](https://www.gradio.app/) and [OpenCV](https://opencv.org/)

---

*Developed as a mini project at University of Visvesvaraya College of Engineering, Bengaluru.*
