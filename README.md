# VectorFit

**Adaptive Singular & Bias Vector Fine-Tuning of Pre-trained Foundation Models**

📄 *Paper*: [VectorFit](https://arxiv.org/abs/2503.19530)
🏫 *Affiliation*: IIT Indore
🧠 *Category*: Parameter-Efficient Fine-Tuning (PEFT)
🎯 *Domains*: Language, Vision, Multimodal, Image Generation

---

## 🔍 Overview

Parameter-Efficient Fine-Tuning (PEFT) methods such as **LoRA**, **Adapters**, and **AdaLoRA** reduce training cost by introducing *new* low-rank parameters on top of frozen pre-trained models. While effective, these methods consistently exhibit a **performance gap** compared to full fine-tuning—especially in **extreme low-parameter regimes**.

**VectorFit** proposes a fundamentally different paradigm.

Instead of learning *new* matrices, VectorFit:

* Directly **adapts existing singular and bias vectors**
* Exploits **transformational structure already encoded** in pre-trained weights
* Produces **high-rank functional updates** using < 0.1% trainable parameters

Across **19 datasets** spanning language, vision, and image generation, VectorFit matches or outperforms state-of-the-art PEFT methods while remaining highly parameter-efficient.

---

## ✨ Key Contributions

* **Singular-Vector Fine-Tuning**
  Fine-tunes singular values and bias vectors instead of introducing new Architecture-level adapters.

* **High-Rank Updates Without Full FT**
  Achieves full-fine-tuning-like behavior with orders of magnitude fewer parameters.

* **Adaptive Vector Freezing (AVF)**
  A novel training mechanism that prevents vector co-adaptation and stabilizes optimization.

* **Extreme Parameter Efficiency**
  Strong performance with ≤ 0.1% trainable parameters.

* **General-Purpose Applicability**
  Works across:

  * NLU / QA / NLG
  * Mathematical reasoning
  * Image classification
  * Subject-driven image generation (DreamBooth)

---

## 🧠 Core Idea (GitHub-Safe Math)

Given a pre-trained weight matrix decomposed using Singular Value Decomposition (SVD):

```
W₀ = U Σ Vᵀ
```

Where:

* `U` and `V` are orthogonal matrices
* `Σ` contains singular values

### VectorFit adapts the model by:

* Keeping `U` and `V` **frozen**
* Training only:

  * Singular value updates `ΔΣ`
  * Bias updates `Δb`

The adapted parameters become:

```
W = U (Σ + ΔΣ) Vᵀ
b = b₀ + Δb
```

This enables **global, high-rank transformations** while training a very small number of parameters.

---

## 🏗 Architecture

![VectorFit Architecture](https://github.com/SuhasHegde25/VectorFit/blob/main/vectorfit_parameterization.png)

### Pipeline

1. Perform SVD on pre-trained weight matrices (one-time)
2. Replace original weights with decomposed representation
3. Train only:

   * Selected singular vectors
   * Bias vectors
4. Control learning dynamics using **Adaptive Vector Freezing**

📌 See the architecture illustration in the paper (Figure 2).

---

## ❄️ Adaptive Vector Freezing (AVF)

Training a small set of vectors can lead to **co-adaptation**, where a subset of vectors dominates optimization.

**Adaptive Vector Freezing (AVF)** addresses this by:

* Measuring per-vector training strength
* Identifying over-optimized vectors
* Temporarily freezing them
* Allowing under-trained vectors to receive gradients

This results in:

* Balanced learning dynamics
* Dropout-like regularization
* Improved stability over random freezing or L1 regularization

---

## 📊 Results

![graph](https://github.com/SuhasHegde25/VectorFit/blob/main/Graphs.png)
Accuracy vs Trainable parameter count for SST2 dataset. VectorFit (labeled as VF for brevity) outperforms baselines with 85% less trainable parameters. The graph highlights that VectorFit is a PEFT method in extremely low parameter regime of <0.1% trainable parameters.

---

### 🔢 Quantitative Results (Main)

<!-- TODO: Add main comparison table -->

**Comparison baselines**

* Full Fine-Tuning
* LoRA
* AdaLoRA
* SVFT
* BitFit

**Metrics**

* Accuracy / F1 (classification)
* ROUGE / BLEU (generation)
* Exact Match (QA)
* Image classification accuracy
* DreamBooth identity similarity

---

### 📈 Parameter Efficiency vs Performance

<!-- TODO: Add plot: performance vs % trainable parameters -->

VectorFit consistently lies on the **Pareto frontier**, particularly below **0.1% trainable parameters**, where other PEFT methods degrade sharply.

---

### 🖼 Qualitative Results

<!-- TODO: Add qualitative figures -->

Planned visualizations:

* Subject-driven image generation
* Identity preservation vs LoRA
* Prompt fidelity comparisons
* Texture and detail retention

---

### 🧪 Ablation Studies

<!-- TODO: Add ablation tables -->

Ablations include:

* With vs without AVF
* Singular-only vs singular + bias
* Attention-only vs full-block adaptation
* Rank analysis of weight updates

---

## ⚙️ Experimental Setup

* **Framework**: PyTorch, HuggingFace Transformers
* **Optimizer**: AdamW
* **Precision**: FP16 / BF16
* **Hardware**:

  * NVIDIA A100 (40GB)
  * Titan XP (efficiency analysis)

### Base Models

* DeBERTa-V3-base
* BART-large
* Gemma-7B
* LLaMA-3-8B
* ViT-base
* Stable Diffusion v1.4

---

## 🧠 Why VectorFit Works

* Singular values encode **directional scaling** across representation subspaces
* Updating them enables **global transformations**, not low-rank perturbations
* Bias adaptation adds **translation flexibility**
* AVF prevents vector dominance and improves convergence

Together, these yield **high-rank functional updates** comparable to full fine-tuning.

---

## ⚠️ Limitations

* Requires one-time SVD preprocessing
* AVF introduces additional hyperparameters
* Upper bound on adaptation capacity is fixed by existing vectors
* Future work may selectively adapt `U` and `V`

---

## 🚀 Applications

* Low-budget LLM fine-tuning
* Multi-task adaptation without model duplication
* Edge and on-device deployment
* Efficient DreamBooth personalization
* Continual learning scenarios

---

## 📚 Citation

```bibtex
@article{Hegde2025VectorFit,
  title   = {VectorFit: Adaptive Singular and Bias Vector Fine-Tuning of Pre-trained Foundation Models},
  author  = {Hegde, Suhas and Kaur, Shilpy and Tiwari, Aruna},
  journal = {arXiv preprint arXiv:2503.19530},
  year    = {2025}
}
```

---

## 🛠 Code Status

🚧 **Code release in progress**

Planned:

* HuggingFace PEFT integration
* AVF scheduler implementation
* Training & evaluation scripts
* DreamBooth examples
* Reproducibility configs
