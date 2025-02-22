Gradient Descent with **Momentum** and **RMSProp** are optimization techniques used to speed up and stabilize learning in neural networks. Let's break them down:  

---

## **1. Gradient Descent with Momentum**  
Gradient Descent can be slow when encountering high curvature or local minima. **Momentum** helps accelerate learning by smoothing updates and reducing oscillations.  

### 🔹 **Concept:**  
- Instead of updating weights purely based on the current gradient, we maintain a moving average of past gradients.  
- This allows the optimizer to build **velocity** in directions where gradients are consistent, helping escape flat regions and reducing zig-zagging in high-curvature areas.  

### 🔹 **Formula:**  
1. Compute the velocity update:

$$
   \[
   v_t = \beta v_{t-1} + (1 - \beta) \cdot \nabla J(\theta)
   \]
$$
   
3. Update weights:
   
$$
   \[
   \theta = \theta - \alpha v_t
   \]
$$

   - **\( v_t \)**: Velocity term (previous updates influence the new update).  
   - **$\( \beta \)**: Momentum term (usually **0.9**).  
   - **\( \alpha \)**: Learning rate.  

### 🔹 **Advantages:**  
✅ Speeds up learning by **reducing oscillations**.  
✅ Helps escape **saddle points** and slow regions.  
✅ Improves convergence in deep networks.  

### 🔹 **Intuition:**  
Think of a ball rolling down a hilly landscape:  
- If there’s **momentum**, the ball picks up speed in the right direction and doesn’t get stuck in small dips.  
- Without momentum, the ball might slow down or oscillate unnecessarily.  

---

## **2. RMSProp (Root Mean Square Propagation)**  
RMSProp helps deal with **vanishing/exploding gradients** and adapts the learning rate dynamically for different parameters.  

### 🔹 **Concept:**  
- Maintains a moving average of **squared gradients** and scales the learning rate accordingly.  
- This helps avoid large updates in steep directions and small updates in flat areas.  

### 🔹 **Formula:**  
1. Compute the exponentially decaying squared gradient:  
   \[
   E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) \cdot (\nabla J(\theta))^2
   \]
2. Update weights:  
   \[
   \theta = \theta - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \cdot \nabla J(\theta)
   \]
   - **\( E[g^2]_t \)**: Moving average of squared gradients.  
   - **\( \beta \)**: Decay factor (usually **0.9**).  
   - **\( \epsilon \)**: Small constant (to prevent division by zero).  
   - **\( \alpha \)**: Learning rate.  

### 🔹 **Advantages:**  
✅ Helps **stabilize** training by adapting learning rates.  
✅ Prevents **oscillations** in steep regions.  
✅ Works well for **non-stationary objectives** (changing loss landscapes).  

### 🔹 **Intuition:**  
Think of **RMSProp** as a self-adjusting learning rate:  
- If a parameter receives **large gradients consistently**, its learning rate shrinks.  
- If a parameter’s gradients are **small**, the learning rate increases.  

---

## **Momentum vs. RMSProp: When to Use What?**  
| Feature         | Momentum | RMSProp |
|----------------|----------|---------|
| Goal          | Faster convergence, less oscillation | Adaptive learning rate |
| Handles Flat Regions | ✅ Yes | ✅ Yes |
| Handles Noisy Gradients | ❌ No | ✅ Yes |
| Works well in Deep Nets | ✅ Yes | ✅ Yes |
| Works well in Non-Stationary Losses | ❌ No | ✅ Yes |

📌 **Best of Both?** **Adam Optimizer** combines both **Momentum** and **RMSProp** for efficient and adaptive learning. 🚀  

---

