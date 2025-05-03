# KL-Divergence-vs-Cross-Entropy-on-CIFAR-10

# 🔍 KL Divergence vs Cross-Entropy on CIFAR-10

**How minimizing KL divergence is mathematically equivalent to minimizing cross-entropy in practice**


---

## 📘 Introduction

KL Divergence is usually used in tasks such as Knowledge Distillation and models such as Variational Autoencoders (VAEs). However, have you ever wondered if it can be used for classification tasks?

KL Divergence and Cross-Entropy are mathematically linked — minimizing one amounts to minimizing the other when using one-hot labels. In this project, I walk through that relationship both theoretically and empirically using CIFAR-10 and PyTorch.

---

## 🧪 What This Repo Covers

- Derivation of the relation between KL Divergence and Cross-Entropy
- PyTorch implementation using:
  - Cross-entropy with hard labels
  - KL divergence with hard labels
  - KL divergence with soft labels (from a pretrained ResNet50 teacher)
- Training plots and test accuracy comparison

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/kl-vs-crossentropy.git
cd kl-vs-crossentropy
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train Teacher (ResNet50 on CIFAR-10)
```bash
python src/train_resnet50.py
```

### 2. Generate Soft Labels from Teacher
```bash
python src/generate_soft_labels.py
```

### 3. Train Student (SmallNet) Under Different Loss Functions

- **Cross-Entropy (Hard Labels)**
  ```bash
  python src/train_smallnet.py --loss crossentropy
  ```
- **KL Divergence (Hard Labels)**
  ```bash
  python src/train_smallnet.py --loss kldiv
  ```
- **KL Divergence (Soft Labels)**
  ```bash
  python src/train_smallnet.py --loss kldiv --use_soft
  ```
  
### 4. Evaluate and Plot

```bash
python src/evaluate.py
```

## 📊 Results

| Model     | Loss Function   | Label Type | Training Time (min) | Final Test Accuracy (%) |
|-----------|------------------|------------|----------------------|--------------------------|
| SmallNet  | CrossEntropy     | Hard       | 4.01                 | 70.68                    |
| SmallNet  | KL Divergence    | Hard       | 4.02                 | 71.61                    |
| SmallNet  | KL Divergence    | Soft       | 4.00                 | 67.78                    |

<p align="center">
  <img src="https://user-images.githubusercontent.com/.../loss_accuracy_curves.png" alt="Training Curves" width="80%">
</p>

## ✅ Conclusion

Through this post, I wanted to show that minimizing KL Divergence is basically the same as minimizing Cross Entropy when we're working with hard labels. The only difference between them is the entropy of the true distribution, H(P), which stays constant during training and doesn't affect gradients. So both losses end up doing the same thing.
When it comes to soft labels, KL Divergence becomes more useful. Since soft labels give a full probability distribution, KL can take advantage of that extra information, which is commonly used in Knowledge Distillation.

Overall, this was a good way to understand both the math and the practical behavior of these two losses. If you've only used Cross Entropy before, hopefully this gives you some insight into when and why KL Divergence might be used.



