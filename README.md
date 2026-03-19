<div align="center">

# 🌍 Ground Motion Quality Assessment via Deep Learning
### Official implementation for the paper: 
**"Generalizability and Explainability of Deep Learning Models to Assess Ground Motion Quality"** *(Namin et al., 202x)*

---

</div>

## 📂 Project Overview
This repository provides two Deep Learning pipelines designed to automate the classification of earthquake ground motion as **"High Quality"** or **"Low Quality,"** eliminating the need for manual expert review in large seismic databases.

---

## 🖼️ 1. Image-Based CNN
<p align="justify">
This approach treats waveform quality as a visual recognition task, mimicking how human experts "scan" plots for anomalies.
</p>

* **Core File:** `G_IM_Model_training.ipynb`
* **Input:** 2D images ($300 \times 1200$ pixels) representing three-component seismic acceleration plots.
* **Architecture:** A 5-layer 2D Convolutional Neural Network (CNN) optimized for high-aspect-ratio seismic imagery.

### ✨ Key Features
* **Class Balancing:** Implements **SMOTE** on image feature vectors to handle the lack of "Low Quality" records.
* **Optimization:** Uses a sequential hyperparameter tuning pipeline for dropout and learning rates.

---

## 📈 2. Time-Series Based Model (ResNet)
<p align="justify">
This approach processes raw numerical data directly to capture complex temporal dependencies.
</p>

* **Core File:** `G_ResNet_Model_training.ipynb`
* **Input:** 1D three-component raw acceleration time histories.
* **Architecture:** A 1D Residual Network (ResNet) with six residual blocks and skip connections.

### ✨ Key Features
* **Waveform Standardization:** Resamples all records to a uniform $0.005s$ time step.
* **Explainability:** Includes **Occlusion Sensitivity Analysis** to visualize which segments (e.g., pre-event noise) influence the model’s decision.

---

## 🛠️ Prerequisites & Setup

### Required Libraries
| Category | Libraries |
| :--- | :--- |
| **Frameworks** | TensorFlow 2.x, Keras, PyTorch |
| **Optimization** | keras-tuner |
| **Data Science** | numpy, pandas, scikit-learn, matplotlib, seaborn |
| **Specialized** | imbalanced-learn (SMOTE), ObsPy |

### Expected Data Format
The notebooks read metadata from pickle files: `tr_No_Fil.pkl`, `val_No_Fil.pkl`, and `te_No_Fil.pkl`.

---

## 🚀 How to Use
1.  **Preprocessing:** Use `restructure_dataframe_3` to format your seismic records.
2.  **Training:** Run the Hyperparameter Tuning cells to find optimal weights.
3.  **Evaluation:** Generate Confusion Matrices and Classification Reports (Precision, Recall, F1-Score).
4.  **Explainability:** Run Occlusion Sensitivity cells in the ResNet notebook for waveform heatmaps.

---

## 📊 Performance Summary
* **ResNet:** High internal accuracy (**~93%**), ideal for consistent training distributions.
* **CNN (Image-based):** Superior **Generalizability (75% accuracy)** on unseen external datasets.

---

## 📝 Citation
```bibtex
@article{namin202x,
  title={Generalizability and Explainability of Deep Learning Models to Assess Ground Motion Quality},
  author={Namin, A. and Kottke, A. and Thompson, E. and Esteghamati, M. Z.},
  journal={Earthquake Spectra},
  year={202x}
}
