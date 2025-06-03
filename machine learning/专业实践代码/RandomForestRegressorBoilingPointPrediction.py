import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # 随机森林模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ─── 构造数据 ─────────────────────────────
data = {
    "MolecularWeight": [18, 46, 60, 74, 88, 100, 112],
    "HBD": [1, 1, 2, 1, 1, 2, 2],
    "HBA": [2, 2, 2, 2, 2, 3, 3],
    "TPSA": [20.2, 46.5, 46.5, 46.5, 46.5, 58.3, 70.1],
    "BoilingPoint": [100, 78, 82, 97, 117, 129, 140]
}
df = pd.DataFrame(data)

# ─── 特征和目标值 ─────────────────────────
X = df[["MolecularWeight", "HBD", "HBA", "TPSA"]]  # 模型输入
y = df["BoilingPoint"]  # 模型输出

# ─── 拆分训练测试集 ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ─── 建立随机森林模型并训练 ────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ─── 预测与评估 ───────────────────────────
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ─── 画图对比预测与真实 ──────────────────
plt.scatter(y_test, y_pred)
plt.xlabel("True Boiling Point")
plt.ylabel("Predicted Boiling Point")
plt.title("Random Forest: Prediction vs True")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray')
plt.grid(True)
plt.show()

# ─── 打印特征重要性 ──────────────────────
importances = model.feature_importances_
features = X.columns
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.3f}")


# Machine Learning and Artificial Intelligence for the Rational Design of Circularly Polarized Room-Temperature Phosphorescent (CPRTP) Materials

## Abstract

Circularly polarized room-temperature phosphorescence (CPRTP) materials have attracted increasing attention due to their promising applications in chiral optoelectronics, anti-counterfeiting, and bioimaging. However, the empirical and time-consuming trial-and-error approach in optimizing their chiroptical performance, particularly the luminescence dissymmetry factor (g\_lum), has significantly limited development efficiency. In this review, we present a comprehensive overview of how machine learning (ML) and artificial intelligence (AI) technologies can be systematically integrated to accelerate the design and optimization of CPRTP materials. From data acquisition to predictive modeling and experimental validation, we propose a complete workflow for ML-assisted CPRTP material discovery.

## 1. Introduction

The design of efficient CPRTP materials requires the precise modulation of multiple parameters such as molecular chirality, emission wavelength, circular dichroism (CD)-photoluminescence (PL) spectral overlap, crosslinking degree, and supramolecular alignment. Traditionally, these factors are optimized individually through heuristic methods, which is labor-intensive and often suboptimal. The emergence of AI and ML provides an opportunity to revolutionize the design process by enabling prediction and optimization based on past data.

## 2. Key Descriptors for CPRTP Performance

The most crucial performance indicator of a CPRTP material is the luminescence dissymmetry factor, g\_lum, which is influenced by several experimentally controllable features:

* **PL peak wavelength (nm)**: emission maxima of phosphorescent molecules
* **CD/PL overlap score**: degree of spectral alignment between CD and PL bands
* **Crosslinker ratio (%)**: proportion of crosslinking agents influencing rigidity and triplet confinement
* **Chiral dopant ratio (%)**: control over handedness and chiral domain size
* **Cholesteric pitch (nm)**: mesophase distance governing macroscopic helical order
* **Film thickness (nm)** and **annealing conditions**: morphology regulation factors

These features serve as input descriptors for supervised learning models aimed at predicting g\_lum.

## 3. Machine Learning Models for g\_lum Prediction

### 3.1 Model Types

Several regression models have shown effectiveness for structure-property prediction:

* **Linear Regression**: baseline model with limited capability to capture nonlinearity
* **Random Forest Regressor**: ensemble tree-based method suitable for small datasets and nonlinear trends
* **XGBoost / LightGBM**: gradient-boosted decision trees with high accuracy and scalability
* **Neural Networks (MLP)**: powerful function approximators for more complex descriptor spaces
* **Gaussian Process Regression (GPR)**: provides uncertainty estimation, suitable for Bayesian optimization

### 3.2 Model Workflow

1. **Data Collection**: Historical experimental records organized into a CSV or database format
2. **Feature Engineering**: Deriving or computing new descriptors (e.g., spectral overlap, molecular rigidity index)
3. **Model Training**: Cross-validated fitting with hyperparameter tuning
4. **Model Evaluation**: R^2 score, MSE, and external validation
5. **Prediction**: Screening untested combinations for high g\_lum

## 4. Optimization Strategy: Discovering Optimal Conditions

Once a reliable model is trained, it can be used to predict g\_lum for hypothetical experimental conditions. Two common strategies for optimization are:

* **Grid Search**: Enumerating all variable combinations within a feasible domain to find the maximum predicted g\_lum
* **Bayesian Optimization**: Iteratively suggesting new experiments that balance exploration and exploitation, guided by uncertainty-aware models like GPR

These approaches allow the identification of optimal conditions such as the ideal PL peak, dopant ratio, or crosslinking level, maximizing chiroptical performance.

## 5. Case Study: Inverse Design of PL Molecules

By fixing other parameters (e.g., chiral dopant ratio, crosslinking degree), the trained model can scan across a library of emission wavelengths (representing different phosphorescent molecules) to identify the peak wavelength that yields the highest g\_lum. This facilitates rational selection of emitters based on predicted chiroptical output, moving towards data-driven inverse design.

## 6. Toward Closed-Loop Experimentation

The ultimate goal of integrating AI with CPRTP material discovery is to establish a closed-loop system, where machine learning models suggest experiments, automated synthesis or robotic platforms execute them, and results are fed back to continuously improve the model. This autonomous design-synthesis-characterization loop represents a paradigm shift in material discovery.

## 7. Conclusion and Outlook

Machine learning and AI offer transformative potential for the rational design of CPRTP materials by reducing trial-and-error, accelerating discovery, and guiding synthetic decisions. As more data becomes available and experimental integration improves, we envision AI-assisted platforms becoming standard tools in chiral luminescent material laboratories. Future efforts should focus on expanding datasets, incorporating molecular structure descriptors (e.g., SMILES, fingerprints), and developing interpretable models that bridge domain knowledge with data-driven prediction.

---

**Keywords**: CPRTP, circularly polarized luminescence, g\_lum, machine learning, inverse design, Bayesian optimization, phosphorescent materials
