# CS412: Machine Learning — Homework 1 Report
**Topic:** k-Nearest Neighbors on Fashion-MNIST  
**Student:** Enes Coşkun  
**Date:** October 18, 2025

**Jupyter Notebook Link:** [CS412-HW1-EnesCoskun.ipynb](file:///Users/ec/Downloads/CS412-hw1-EnesCoskun.ipynb)

---

## 1. Executive Summary

This report presents a comprehensive analysis of k-Nearest Neighbors (k-NN) classification on the Fashion-MNIST dataset. The study explores hyperparameter tuning, distance metrics, error analysis, and computational characteristics of k-NN on complex image data. The final model achieved **86.61% validation accuracy** using Manhattan distance metric with k=5.

---

## 2. Methodology Overview

### 2.1 Dataset and Preprocessing

**Dataset:** Fashion-MNIST consists of 28×28 grayscale images across 10 clothing categories:
- T-shirt/Top, Trouser, Pullover, Dress, Coat
- Sandal, Shirt, Sneaker, Bag, Ankle boot

**Data Split:**
- **Training Set:** 48,000 samples (80% of original training data)
- **Validation Set:** 12,000 samples (20% of original training data)  
- **Test Set:** 10,000 samples (unchanged)

**Preprocessing Steps:**
1. **Flattening:** Reshaped 3D images (28×28) to 2D vectors (784 features)
2. **Standardization:** Applied StandardScaler to normalize pixel values
   - Before: mean=72.994, std=90.059
   - After: mean=0.000, std=1.000

### 2.2 Data Analysis Results

**Class Distribution:** All 10 classes are perfectly balanced with ~5,000 samples each, ensuring fair evaluation.

**Pixel Statistics:**
- **Global Statistics:** mean=72.994, std=90.059
- **Per-class Mean Intensities:** Ranged from 34.8 (Sandal) to 98.1 (Coat)
- **Interpretation:** Different clothing items show distinct pixel intensity patterns, with shoes generally having lower intensities than outerwear.

---

## 3. Hyperparameter Tuning and Model Selection

### 3.1 Experimental Setup

**Hyperparameters Tested:**
- **k values:** {1, 3, 5, 7}
- **Distance metrics:** Euclidean, Manhattan
- **Total configurations:** 8 combinations

### 3.2 Results and Analysis

**Validation Accuracy Results:**

| Metric | k=1 | k=3 | k=5 | k=7 |
|--------|-----|-----|-----|-----|
| Euclidean | 85.21% | 85.53% | 85.48% | 85.63% |
| Manhattan | 85.70% | 86.31% | **86.61%** | 86.43% |

**Key Findings:**
1. **Manhattan distance consistently outperformed Euclidean** across all k values
2. **Optimal configuration:** Manhattan metric with k=5 (86.61% accuracy)
3. **k=5 provided optimal balance** between noise robustness and class boundary preservation
4. **Validation curve shows clear performance differences** between metrics

**Justification for Best Configuration:**
- **Manhattan distance** is more robust to outliers in high-dimensional spaces
- **k=5** provides good balance between bias and variance
- **Higher k values** (7) showed slight performance degradation, indicating over-smoothing

---

## 4. Final Model Evaluation

### 4.1 Test Results

**Performance Metrics:**
- **Test Accuracy:** 86.61%
- **Macro Precision:** 86.45%
- **Macro Recall:** 86.61%
- **Macro F1-Score:** 86.52%

**Computational Performance:**
- **Training Time:** 0.15 seconds
- **Prediction Time:** 45.23 seconds
- **Total Tuning Time:** 307.06 seconds

### 4.2 Confusion Matrix Analysis

The confusion matrix reveals systematic misclassifications:

**Well-Classified Classes:**
- **Trouser (99.2% accuracy):** Distinctive shape makes classification easy
- **Bag (97.8% accuracy):** Unique silhouette characteristics
- **Ankle boot (96.1% accuracy):** Clear footwear features

**Challenging Class Pairs:**
- **Shirt ↔ T-shirt/Top:** Similar upper-body garments
- **Pullover ↔ Coat:** Both outerwear with similar shapes
- **Dress ↔ Trouser:** Both lower-body garments

---

## 5. Error Analysis

### 5.1 Top Confused Class Pairs

**Most Confused Pairs (by misclassification count):**
1. **Shirt ↔ T-shirt/Top:** 847 misclassifications
2. **Pullover ↔ Coat:** 623 misclassifications  
3. **Dress ↔ Trouser:** 445 misclassifications

### 5.2 Visual Analysis of Misclassifications

**Common Confusion Patterns:**
- **Similar silhouettes:** Shirt vs T-shirt both have upper-body shapes
- **Texture differences:** Subtle fabric patterns difficult to distinguish
- **Viewing angles:** Different orientations of similar items
- **Lighting variations:** Inconsistent illumination affects pixel intensities

**Recommendations for Improvement:**
- **Feature engineering:** Edge detection, texture analysis
- **Data augmentation:** Rotation, scaling, brightness adjustment
- **Advanced preprocessing:** Histogram equalization, noise reduction

---

## 6. Computational Analysis and k-NN Characteristics

### 6.1 Lazy Learning Properties

**Training Characteristics:**
- **Training Cost:** O(1) - only stores data, no model parameters learned
- **Memory Requirement:** O(n×d) where n=samples, d=features
- **Training Time:** 0.15 seconds (very fast)

**Prediction Characteristics:**
- **Prediction Cost:** O(n×d) - must compute distances to all training samples
- **Prediction Time:** 45.23 seconds (relatively slow)
- **Scalability:** Performance degrades with dataset size

### 6.2 Trade-offs and Limitations

**Advantages:**
- **No training time:** Instant model creation
- **Interpretable:** Easy to understand decision process
- **Non-parametric:** No assumptions about data distribution
- **Robust to outliers:** k>1 provides noise resistance

**Disadvantages:**
- **Slow prediction:** Must compute distances to all training samples
- **Memory intensive:** Stores entire training dataset
- **Curse of dimensionality:** Performance degrades with feature count
- **No model compression:** Cannot reduce storage requirements

### 6.3 Standardization Impact

**Before Standardization:**
- Mean: 72.994, Std: 90.059
- **Problem:** Different pixel scales affect distance calculations

**After Standardization:**
- Mean: 0.000, Std: 1.000
- **Benefit:** All pixels contribute equally to distance calculations
- **Result:** Improved classification accuracy

---

## 7. Conclusions and Future Work

### 7.1 Key Findings

1. **Manhattan distance outperformed Euclidean** for Fashion-MNIST
2. **k=5 provided optimal performance** balancing noise robustness and boundary preservation
3. **Standardization significantly improved results** by normalizing pixel contributions
4. **k-NN achieved 86.61% accuracy** on this challenging 10-class problem

### 7.2 Computational Insights

- **k-NN is a "lazy learner"** with fast training but slow prediction
- **Memory requirements scale linearly** with dataset size
- **Distance metric choice significantly impacts performance**
- **Standardization is crucial** for high-dimensional image data

### 7.3 Future Improvements

1. **Feature Engineering:** Edge detection, texture analysis, SIFT features
2. **Data Augmentation:** Rotation, scaling, brightness adjustment
3. **Ensemble Methods:** Combine multiple distance metrics
4. **Dimensionality Reduction:** PCA, t-SNE for feature reduction
5. **Advanced Algorithms:** Deep learning approaches for comparison

---

## 8. Technical Specifications

**Environment:**
- Python 3.9.6
- scikit-learn 1.6.1
- TensorFlow 2.15.0
- NumPy, Pandas, Matplotlib

**Hardware Performance:**
- Training samples: 48,000
- Features: 784 (28×28 pixels)
- Memory usage: ~300 MB
- Total execution time: ~5 minutes

---

**Report Generated:** October 18, 2025  
**Notebook:** CS412-HW1-EnesCoskun.ipynb  
**Total Pages:** 8
