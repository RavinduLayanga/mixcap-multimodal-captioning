# MixCap: Multimodal Video Captioning with Dual-Target MixUp 🎥 🎵

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter\&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Research_Complete-success)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **A novel approach to video captioning that fuses audio (Wav2Vec2) and visual (BLIP-2) features, utilizing a custom "Dual-Target MixUp" augmentation strategy to solve overfitting on low-resource datasets.**

---

## Overview

**MixCap** is a deep learning research project designed to generate accurate natural language descriptions for videos. While traditional models often rely solely on visual data, I designed MixCap to process **both sight and sound** to understand context (e.g., hearing a car engine helps identify a car even if the video is blurry).

To address the challenge of training on limited datasets (like MSR-VTT), this project introduces a **Dual-Target MixUp Strategy** applied at the *feature level*, significantly improving generalization without breaking the temporal sequence of the video.

### Live Demo & Application

This repository contains the **research code, model architecture, and training pipeline**.
To see the deployed model in a user-friendly Web Application (React + Flask), check out the **MixCap Web Platform**:

**[View the Web App Repository Here](https://github.com/RavinduLayanga/mixcap-web-platform)**

---

## Key Innovation: Dual-Target MixUp

Standard data augmentation (like flipping images) doesn't work well for video sequences or fused audio-visual data.

**The Solution:**
I implemented a custom **Dual-Target MixUp** strategy that operates on the **fused feature space** (2432-dimensional vectors).

1. **Feature Mixing:** Linearly interpolates between the audio-visual features of Video A and Video B.
2. **Label Mixing:** Simultaneously mixes the target caption embeddings.
3. **Result:** The model learns to predict "between" concepts, preventing memorization of the training data.

---

## Evaluation Results

The model was evaluated on two standard MSR-VTT splits: the **1k-A Split** (standard academic benchmark) and the **Full Test Set** (2,990 videos) to test robustness.

| Metric     | 1k-A Split (Benchmark) | Full Test Set (Robustness) |
| :--------- | :--------------------: | :------------------------: |
| **BLEU-4** |        **0.43**        |          **0.44**          |
| **CIDEr**  |        **0.53**        |          **0.55**          |
| BLEU-1     |          0.84          |            0.84            |
| BLEU-2     |          0.71          |            0.71            |
| BLEU-3     |          0.57          |            0.57            |
| ROUGE-L    |          0.62          |            0.63            |
| METEOR     |          0.29          |            0.30            |

> **Note:** The 1k-A split results are provided for direct comparison with state-of-the-art methods found in literature.

---

## Architecture & Tech Stack

The architecture follows an Encoder-Decoder design:

* **Visual Encoder:** **BLIP-2** (frozen) extracts high-level semantic visual features.
* **Audio Encoder:** **Wav2Vec2** (frozen) extracts audio signal features.
* **Fusion Layer:** Concatenates features into a unified 2432D representation.
* **Decoder:** A custom Transformer with **Bidirectional Cross-Attention**.
* **Tokenizer:** SentencePiece (BPE) for efficient text generation.

**Libraries:** `PyTorch`, `Transformers (Hugging Face)`, `Pandas`, `NumPy`, `Scikit-Learn`.

---

## Repository Structure

```bash
├── features_extraction/
│   ├── mixcap-video-feature-extract.ipynb  # BLIP-2 extraction script
│   ├── mixcap-audio-feature-extract.ipynb  # Wav2Vec2 extraction script
├── model/
│   ├── mixcap-final-best-model.ipynb       # Core model architecture & training loop
│   ├── spm_tokanizer.ipynb                 # SentencePiece tokenizer training
├── evaluation/
│   ├── mixcap-final-test-1ka.ipynb         # Inference on Benchmark Split
│   ├── mixcap-final-test-full.ipynb        # Inference on Full Test Set
├── docs/
│   ├── dissertation.pdf                    # Full research paper
└── README.md
```

---

## Usage & Reproduction

### 1. Prerequisites

Ensure you have a GPU-enabled environment (NVIDIA CUDA recommended).

```bash
pip install torch transformers pandas sentencepiece
```

### 2. Feature Extraction

**Note:** This project automatically fetches pre-trained models. It does not include the 10GB BLIP-2 model in the repo.

Run the notebooks in `feature_extraction/`.

The script will automatically download BLIP-2 and Wav2Vec2 from Hugging Face.

This generates the `.npy` feature files required for training.

### 3. Training

Open `model/mixcap-final-best-model.ipynb`:

* Point the dataloader to your extracted features.
* Run the training cells. The Dual-MixUp augmentation is applied automatically within the training loop.

---

## Acknowledgements & Citations

This project builds upon the following datasets and pre-trained models.

### Dataset & Splits

**MSR-VTT (Original Dataset):**

Xu, J., Mei, T., Yao, T., & Rui, Y. (2016). *MSR-VTT: A Large Video Description Dataset for Bridging Video and Language.*

[Data Source](https://www.kaggle.com/datasets/nyhuka/msrvtt)

**1k-A Split (JSFusion Split):**

The project uses the specific 1,000-video test split introduced by Yu et al., which is the standard benchmark for this task.

Yu, Y., Kim, J., & Kim, G. (2018). *A Joint Sequence Fusion Model for Video Question Answering and Retrieval.* ECCV.

---

### Pre-trained Models

State-of-the-art pre-trained models are utilized for feature extraction. These models are downloaded automatically via the Hugging Face transformers library.

**Visual Encoder:** BLIP-2 (Salesforce)

Li, J., et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training.*

[Hugging Face Model Card](https://huggingface.co/Salesforce/blip2-opt-2.7b)

**Audio Encoder:** Wav2Vec2 (Meta AI)

Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning.*

[Hugging Face Model Card](https://huggingface.co/facebook/wav2vec2-base-960h)

---

## Author

* **Ravindu Layanga**  
* *BSc (Hons) Computer Science*  
* University of Westminster / Informatics Institute of Technology (IIT)
