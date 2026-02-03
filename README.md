---

## VectorFit

**Adaptive Singular & Bias Vector Fine-Tuning of Pre-trained Foundation Models**

📄 *Paper*: VectorFit (arXiv:2503.19530)
🏫 *Affiliation*: IIT Indore
🧠 *Category*: Parameter-Efficient Fine-Tuning (PEFT)
🎯 *Domains*: Language, Vision, Multimodal, Image Generation

---

## 🔍 Overview

Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA, Adapters, and AdaLoRA reduce training cost by introducing *new* low-rank or sparse parameters alongside frozen pre-trained weights. While effective, these methods suffer from a persistent **performance gap** compared to full fine-tuning—especially in **extremely low-budget regimes**.

**VectorFit** takes a fundamentally different approach.

Instead of adding new trainable matrices, VectorFit:

* **Directly adapts the singular vectors and biases of pre-trained weight matrices**
* Exploits the **structural and transformational knowledge already embedded** in the weights
* Produces **high-rank incremental updates** comparable to full fine-tuning

With **≤ 0.1% trainable parameters**, VectorFit consistently matches or surpasses state-of-the-art PEFT methods across **19 datasets** spanning language, vision, and generative tasks.

---

## ✨ Key Contributions

* **Singular-Vector-Based Fine-Tuning**
  Fine-tunes singular values (Σ) and bias vectors instead of learning new weights.

* **High-Rank Adaptation Without Full-FT**
  Achieves full-rank–like updates using a fraction of parameters.

* **Adaptive Vector Freezing (AVF)**
  A novel mechanism that balances training across singular and bias vectors, preventing co-adaptation.

* **Extreme Parameter Efficiency**
  Outperforms baselines with up to **9× fewer trainable parameters**.

* **General-Purpose PEFT**
  Works across:

  * NLU, QA, NLG
  * Image classification
  * Subject-driven image generation (DreamBooth)

---

## 🧠 Core Idea

Given a pre-trained weight matrix:

[
W_0 = U \Sigma V^T
]

VectorFit:

* Keeps **U** and **V** frozen
* Trains:

  * the **singular vector Σ**
  * the **bias vector b**

The adapted weight becomes:

[
W = U (\Sigma + \Delta \Sigma) V^T,\quad b = b_0 + \Delta b
]

This allows **high-rank incremental updates** without introducing new parameter matrices.

---

## 🏗 Architecture

![VectorFit Architecture](assets/vectorfit_architecture.png)

**Pipeline**:

1. Perform SVD on pre-trained weights (once, before fine-tuning)
2. Replace original weights with decomposed form
3. Train only:

   * selected singular vectors
   * bias vectors
4. Control training dynamics via **Adaptive Vector Freezing**

> See Figure 2 (page 3) of the paper for the full architecture diagram .

---

## ❄️ Adaptive Vector Freezing (AVF)

Training only a small number of vectors can lead to **co-adaptation**, where a few vectors dominate optimization.

AVF solves this by:

* Measuring **training strength** of each vector
* Periodically **freezing the most over-trained vectors**
* Allowing under-trained vectors to catch up

This yields:

* Dropout-like regularization effects
* More balanced optimization
* Better stability than random freezing or L1 regularization

---

## 📊 Results

> **Note:** Below are structured placeholders so you can add results incrementally.

### 🔢 Quantitative Results (Main)

<!-- TODO: Add main results table comparing VectorFit with LoRA, AdaLoRA, SVFT, Full-FT -->

**Covered tasks:**

* GLUE (NLU)
* SQuAD v1.1 / v2.0 (QA)
* XSum, CNN/DailyMail (NLG)
* GSM8K, MATH (Reasoning)
* CIFAR10, GTSRB, MNIST, RESISC45 (Vision)
* DreamBooth (Image Generation)

---

### 📈 Parameter Efficiency vs Performance

<!-- TODO: Add plot: Accuracy / ROUGE / F1 vs % trainable parameters -->

VectorFit consistently lies on the **Pareto frontier**, especially below **0.1% trainable parameters**.

---

### 🖼 Qualitative Results

<!-- TODO: Add image grids for DreamBooth and Stable Diffusion -->

Examples:

* Subject-driven generation
* Prompt fidelity comparisons
* Texture & identity preservation

(See Figure 12 in the paper for reference)

---

### 🧪 Ablation Studies

<!-- TODO: Add ablation tables -->

Included ablations:

* With vs without AVF
* Singular vectors only vs singular + bias
* Attention-only vs full-block adaptation
* Rank analysis of ΔW

---

## ⚙️ Experimental Setup

* **Framework**: PyTorch + HuggingFace Transformers
* **Optimizer**: AdamW
* **Hardware**:

  * NVIDIA A100 (40GB)
  * Titan XP (training speed analysis)

### Base Models

* DeBERTa-V3-base
* BART-large
* Gemma-7B
* LLaMA-3-8B
* ViT-base
* Stable Diffusion v1.4

---

## 🧠 Why VectorFit Works

* Singular values encode **directional scaling** in high-dimensional space
* Updating them enables **global transformations**, not just low-rank perturbations
* Bias training adds **translational freedom**
* AVF ensures **balanced learning dynamics**

Together, these produce **high-rank updates** that closely track Full-FT behavior.

---

## ⚠️ Limitations

* AVF introduces additional hyperparameters
* Upper bound on trainable parameters is fixed (no new matrices)
* Future extensions may involve adapting **U** and **V** selectively

---

## 🚀 Applications

* Low-budget fine-tuning of LLMs
* Multi-task adaptation without model duplication
* On-device / edge deployment
* Efficient DreamBooth-style personalization

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
* Training scripts
* AVF scheduler implementation
* Reproducibility configs
* DreamBooth examples

---
