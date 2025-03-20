# Gradient Descent vs. Stochastic Gradient Descent (SGD)

## 1. **Definition**
| Method | Description |
|--------|------------|
| **Gradient Descent (GD)** | Uses the entire dataset to compute the gradient and update the model parameters. |
| **Stochastic Gradient Descent (SGD)** | Updates the model parameters using a single randomly chosen data point (or a small batch). |

## 2. **Key Differences**
| Aspect | Gradient Descent (GD) | Stochastic Gradient Descent (SGD) |
|--------|-----------------------|----------------------------------|
| **Update Frequency** | Updates parameters after computing gradients on the entire dataset. | Updates parameters after each training example. |
| **Computation Cost** | Expensive for large datasets (slow). | Faster per step but noisier. |
| **Convergence Stability** | More stable, but can get stuck in local minima. | Noisy updates help escape local minima. |
| **Memory Usage** | Requires storing entire dataset. | Uses less memory since it processes one sample at a time. |
| **Speed** | Slower due to full dataset processing. | Faster but may fluctuate more. |
| **Final Solution** | More stable but may converge slowly. | Can oscillate around the optimal solution. |

## 3. **Mathematical Representation**
### **Gradient Descent (Batch Gradient Descent)**
$$
\[
\theta = \theta - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla J(\theta, x_i)
\]$$
- Uses the full dataset (\( m \) samples) to compute the gradient.

### **Stochastic Gradient Descent (SGD)**
$$\[
\theta = \theta - \alpha \cdot \nabla J(\theta, x_i)
\]$$
- Uses only **one** random sample (\( x_i \)) at a time to update the parameters.

## 4. **Variants of Gradient Descent**
- **Batch Gradient Descent** → Uses the full dataset.
- **Stochastic Gradient Descent (SGD)** → Uses one data point at a time.
- **Mini-Batch Gradient Descent** → Uses a small batch of data (common in deep learning).

## 5. **When to Use What?**
| Scenario | Best Choice |
|----------|------------|
| **Small datasets** | **Gradient Descent (GD)** (stable and accurate). |
| **Large datasets** | **SGD** (faster and memory-efficient). |
| **Deep learning** | **Mini-batch GD** (balances speed and stability). |
| **Avoiding local minima** | **SGD** (randomness helps escape bad local optima). |

## 6. **Final Thoughts**
- **GD is best for convex and smooth optimization problems** where stability is key.
- **SGD is better for large-scale problems (deep learning, online learning)** due to its efficiency.
- **Mini-batch Gradient Descent** is a compromise between GD and SGD, often used in deep learning.



# Gradient Descent: Momentum & RMSProp

## 1. Gradient Descent with Momentum  
Gradient Descent with Momentum helps accelerate learning by smoothing updates and reducing oscillations.

### 🔹 Formula:
$$v_t = \beta v_{t-1} + (1 - \beta) \cdot \nabla J(\theta)$$

$$\theta = \theta - \alpha v_t$$

### 🔹 Explanation:
- **v_t**: Velocity term (tracks past gradients).  
- **$\beta$**: Momentum coefficient (typically **0.9**).  
- **$\alpha$**: Learning rate.  

### 🔹 Intuition:
Think of **momentum** like rolling a ball down a hill—it picks up speed in the right direction, reducing oscillations.
---

## 2. RMSProp (Root Mean Square Propagation)  
RMSProp adjusts the learning rate dynamically for different parameters.

### 🔹 Formula:
$$ v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta) $$

$$\theta = \theta - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \cdot \nabla J(\theta)$$

### 🔹 Explanation:
- **$E[g^2]_t$**: Moving average of squared gradients.  
- **$\beta$**: Decay factor (typically **0.9**).  
- **$\epsilon$**: Small constant to prevent division by zero.  

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

