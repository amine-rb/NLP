# Section 2 — Answers

## Question 1 — Geometric Interpretation

### 1a — For c ∈ C⁺
We want to **maximize** σ(cᵀw), pushing it towards 1. This minimizes the loss term −log(σ(cᵀw)).

### 1b — For c ∈ C⁻
We want to **minimize** σ(cᵀw), pushing it towards 0. This minimizes the loss term −log(1 − σ(cᵀw)).

### 1c — Geometric Interpretation

The similarity score σ(cᵀw) depends on the dot product, which can be decomposed as:

$$c^T w = \|c\| \cdot \|w\| \cdot \cos(\theta)$$

where θ is the angle between the two vectors. Therefore:

| Configuration | cos(θ) | σ(cᵀw) |
|---|---|---|
| Aligned vectors (θ ≈ 0°) | ≈ +1 | close to 1 |
| Orthogonal vectors (θ = 90°) | = 0 | ≈ 0.5 |
| Opposite vectors (θ = 180°) | ≈ −1 | close to 0 |

Minimizing the loss is geometrically equivalent to **pulling** the word vector w closer to its positive context words C⁺, and **pushing** it away from the negative context words C⁻. The model thus learns a vector space geometry where semantic proximity is reflected by geometric proximity.

---

## Question 2 — Link with Chopra, Hadsell & LeCun (2005)

### Contrastive Learning (from the introduction)

The core idea of contrastive learning is to learn an **embedding function** such that similar pairs are mapped close together in the learned space, while dissimilar pairs are pushed far apart. In the paper, this is applied to face verification: two photos of the same person should be close, two photos of different people should be distant.

The loss function from the paper is:

$$L(W,(Y, X_1, X_2)^i) = (1 − Y) L_G(E_W(X_1, X_2)^i) + Y L_I(E_W(X_1, X_2)^i)$$

### 2a — Analog of Y

Y is a binary variable indicating whether a pair is similar (Y = 0) or dissimilar (Y = 1). In Word2Vec, its analog is the indicator **1_{c∈C⁺}** and **1_{c∈C⁻}**: a positive context word plays the role of a genuine pair (Y = 0), while a negative context word plays the role of an impostor pair (Y = 1).

### 2b — Analog of E_W

E_W is the function that encodes both inputs X₁ and X₂ and computes a distance between them. In Word2Vec, its analog is the **dot product** σ(cᵀw) between the two embeddings. However, they differ in an important way: in the paper, E_W uses the **same network** (Siamese architecture) to encode both X₁ and X₂, whereas Word2Vec uses **two distinct embedding tables** E_w and E_C to encode the target word w and the context word c respectively.

### 2c — Analogs of L_G and L_I

| Paper | Role | Word2Vec analog |
|---|---|---|
| **L_G** (Genuine loss) | Penalizes similar pairs that are too far apart | **−log(σ(cᵀw))** for c ∈ C⁺ |
| **L_I** (Impostor loss) | Penalizes dissimilar pairs that are too close | **−log(1 − σ(cᵀw))** for c ∈ C⁻ |

Both formulations share the same contrastive objective: attract similar pairs and repel dissimilar ones.