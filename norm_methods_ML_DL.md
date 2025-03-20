# Major Normalization Methods in Deep Learning and Machine Learning

Normalization is a crucial step in preparing training datasets for deep learning and machine learning models. It helps improve convergence speed, stability, and overall model performance. Below are some of the major normalization techniques:

## 1. **Min-Max Scaling**
   - Formula:  
     $$\[
     X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
     \]$$
   - Scales features to a fixed range, typically [0, 1] or [-1, 1].
   - Sensitive to outliers.

## 2. **Z-Score Normalization (Standardization)**
   - Formula:  
     $$\[
     X' = \frac{X - \mu}{\sigma}
     \]$$
   - Centers the data around zero with unit variance.
   - Useful for algorithms assuming normally distributed data.

## 3. **Robust Scaling**
   - Formula:  
     $$\[
     X' = \frac{X - \text{median}(X)}{\text{IQR}(X)}
     \]$$
   - Uses median and interquartile range (IQR) instead of mean and standard deviation.
   - Effective when handling outliers.

## 4. **Log Transformation**
   - Formula:  
     $$\[
     X' = \log(X + \epsilon)
     \]$$
   - Helps handle skewed data by reducing the impact of large values.
   - Useful for datasets following a power-law distribution.

## 5. **Batch Normalization (For Deep Learning)**
   - Normalizes activations within a neural network during training.
   - Helps stabilize and accelerate training by reducing internal covariate shift.

## 6. **Layer Normalization**
   - Similar to batch normalization but normalizes across features for each individual data point.
   - Effective for recurrent neural networks (RNNs).

## 7. **Instance Normalization**
   - Normalizes each sample independently over spatial dimensions.
   - Common in image processing tasks like style transfer.

## 8. **Group Normalization**
   - Divides channels into groups and normalizes each group separately.
   - Useful for small batch sizes where batch normalization is ineffective.

## Conclusion
Choosing the right normalization method depends on the type of data, distribution, and specific machine learning task. In deep learning, **batch normalization** is widely used for neural networks, while **min-max scaling** and **Z-score normalization** are popular for standard machine learning models.

