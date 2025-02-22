# Gradient Descent with **Momentum** and **RMSProp**

Optimization techniques used to speed up and stabilize learning in neural networks:

---

## **1. Gradient Descent with Momentum**  
Accelerates learning by smoothing updates and reducing oscillations.

### 🔹 **Formula**  
1. Compute velocity update:
$$
v_t = \beta v_{t-1} + (1 - \beta) \cdot \nabla J(\theta)
$$

2. Update weights:
$$
\theta = \theta - \alpha v_t
$$

- **\( v_t \)**: Velocity term.  
- **\( \beta \)**: Momentum term (typically **0.9**).  
- **\( \alpha \)**: Learning rate.  

- **$v_t$**: Velocity term.
- **$\beta$**: Momentum term (typically **0.9**).
- **$\alpha$**: Learning rate.

---

## **2. RMSProp (Root Mean Square Propagation)**  
Adapts learning rates dynamically to stabilize training.

### 🔹 **Formula**  
1. Compute squared gradient average:
$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) \cdot (\nabla J(\theta))^2
$$

2. Update weights:
$$
\theta = \theta - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \cdot \nabla J(\theta)
$$

- **\( E[g^2]_t \)**: Moving average of squared gradients.  
- **\( \beta \)**: Decay factor (typically **0.9**).  
- **\( \epsilon \)**: Small constant (e.g., \(10^{-8}\)) to avoid division by zero.  

---

## **Momentum vs. RMSProp**  
| Feature         | Momentum | RMSProp |
|----------------|----------|---------|
| Goal           | Reduce oscillations | Adapt learning rates |
| Handles Flat Regions | ✅ | ✅ |
| Handles Noisy Gradients | ❌ | ✅ |
| Works in Non-Stationary Losses | ❌ | ✅ |

📌 **Best of Both Worlds**: Use **Adam Optimizer**, which combines Momentum and RMSProp.  

---

### Key Fixes:  
1. Removed redundant `\[...\]` inside `$$...$$` blocks.  
2. Fixed LaTeX alignment and spacing.  
3. Corrected formula numbering (1 → 2 instead of 1 → 3).  

**To render equations on GitHub**, install a browser extension like [MathJax for GitHub](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima).  
