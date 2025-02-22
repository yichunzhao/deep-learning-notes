# Gradient Descent: Momentum & RMSProp

## 1. Gradient Descent with Momentum  
Gradient Descent with Momentum helps accelerate learning by smoothing updates and reducing oscillations.

### 🔹 Formula:
$$
v_t = \beta v_{t-1} + (1 - \beta) \cdot \nabla J(\theta)
$$

$$
\theta = \theta - \alpha v_t
$$

### 🔹 Explanation:
- **v_tv_t**: Velocity term (tracks past gradients).  
- **\beta\beta**: Momentum coefficient (typically **0.9**).  
- **\alpha\alpha**: Learning rate.  

### 🔹 Intuition:
Think of **momentum** like rolling a ball down a hill—it picks up speed in the right direction, reducing oscillations.
---

## 2. RMSProp (Root Mean Square Propagation)  
RMSProp adjusts the learning rate dynamically for different parameters.

### 🔹 Formula:

$$ v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta) $$

```latex
$$ v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta) $$
```

$$
\theta = \theta - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \cdot \nabla J(\theta)
$$

### 🔹 Explanation:
- **$E[g^2]_t$**: Moving average of squared gradients.  
- **\beta\beta**: Decay factor (typically **0.9**).  
- **\epsilon\epsilon**: Small constant to prevent division by zero.  

### 🔹 Intuition:
RMSProp **reduces oscillations** and adapts learning rates dynamically, making it ideal for deep learning.

---

## 🚀 Momentum vs. RMSProp
| Feature        | Momentum | RMSProp |
|---------------|---------|---------|
| Goal         | Faster convergence | Adaptive learning rate |
| Handles Flat Regions | ✅ Yes | ✅ Yes |
| Handles Noisy Gradients | ❌ No | ✅ Yes |
| Deep Learning Friendly | ✅ Yes | ✅ Yes |

💡 **Bonus Tip:** The **Adam optimizer** combines both **Momentum** and **RMSProp** for better performance! 🚀

---

