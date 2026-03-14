# Brain Tumor Signal Classification Project

This project leverages advanced machine learning and deep learning techniques to classify biological signals (related to brain tumor diagnostics) into five distinct categories. The implementation transitions from a standard neural network approach to a high-performance stacked ensemble model.

## 1. Project Overview
The objective is to accurately categorize diagnostic signal data. The project emphasizes a comparative approach, evaluating the performance of **Artificial Neural Networks (ANN)** against **Gradient Boosted Decision Trees (GBDT)** and **Ensemble Stacking**.

## 2. Dataset Characteristics
- **Volume**: 11,500 samples, providing a robust basis for training and validation.
- **Dimensionality**: 178 numerical features (labeled `X1` through `X178`), likely representing digitized signal amplitudes or transformed spectral data.
- **Target Variable**: 5 balanced classes (1-5), with exactly 2,300 samples per class, ensuring no majority class bias.

## 3. Technical Methodology

### A. Data Preprocessing
- **Standardization**: Uses `StandardScaler` to normalize feature distributions (mean=0, variance=1), which is critical for the stability of neural network optimization.
- **Dimensionality Reduction (PCA)**: Principal Component Analysis is applied to reduce the feature set to 50 components. This helps in filtering noise and reducing the "Curse of Dimensionality" while retaining maximum variance.
- **Feature Selection**: Recursive selection of the top 100 features based on importance scores from ensemble models.

### B. Model Architectures
- **Multi-Layer Perceptron (ANN)**:
    - Architecture: 128 -> 64 -> 32 nodes.
    - Regularization: **Dropout (0.3)** and **L2 (weight decay)** are used to mitigate overfitting.
    - Activation: **ReLU** for hidden layers and **Softmax** for the 5-class output layer.
- **CatBoost & LightGBM**: Optimized gradient boosting models that specialize in handling complex interactions within the high-dimensional signal data.
- **Stacked Ensemble**: A "StackingClassifier" that uses CatBoost and LGBM as base estimators. A final **Logistic Regression** layer is used to aggregate their outputs, achieving a refined accuracy of ~74%.

## 4. Evaluation & Results
- **Feature Importance**: The project includes visualization of which signal components (X-features) are most predictive, allowing for potential biological interpretation.
- **Performance**: The combination of feature selection and ensemble stacking significantly outperformed the initial neural network benchmarks.
- **Confusion Matrix**: Used to validate balanced predictive power across all 5 classes.

---
*Developed using Python (TensorFlow/Keras, Scikit-learn, CatBoost, LightGBM).*
