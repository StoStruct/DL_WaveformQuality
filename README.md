Ground Motion Quality Assessment via Deep Learning
This repository provides the official implementation for the paper: "Generalizability and Explainability of Deep Learning Models to Assess Ground Motion Quality" (Namin et al., 2025).

It features two distinct Deep Learning (DL) pipelines designed to automate the classification of earthquake ground motion (GM) waveforms into "High Quality" or "Low Quality," addressing the scalability challenges of manual expert review in large seismic databases.

📂 Repository Structure
The project is divided into two main methodologies:

1. IM_based/ (Image-Based CNN)
This approach treats waveform quality as a visual recognition task, mimicking how human experts "scan" plots for anomalies.

Core File: G_IM_Model_training.ipynb

Input: 2D images (300x1200 pixels) representing three-component seismic acceleration plots.

Architecture: A 5-layer 2D Convolutional Neural Network (CNN) specifically optimized for high-aspect-ratio seismic imagery.

Key Features:

Data Preprocessing: Inverts plot colors (white-on-black) to emphasize signal features and applies normalization.

Class Balancing: Implements SMOTE (Synthetic Minority Over-sampling Technique) on image feature vectors to handle the scarcity of "Low Quality" records.

Optimization: Uses a sequential hyperparameter tuning pipeline for dropout rates and learning rates to maximize generalizability.

2. ResNet_Model/ (Time-Series ResNet)
This approach processes the raw numerical data directly to capture complex temporal dependencies.

Core File: G_ResNet_Model_training.ipynb

Input: 1D three-component raw acceleration time histories.

Architecture: A 1D Residual Network (ResNet) featuring six residual blocks with skip connections to prevent vanishing gradients in high-dimensional seismic data.

Key Features:

Waveform Standardization: Resamples all records to a uniform 0.005s time step and utilizes zero-padding for length consistency.

Advanced Tuning: Employs the Hyperband optimization algorithm via keras-tuner to find the ideal number of filters and kernel sizes.

Explainability: Includes scripts for Occlusion Sensitivity Analysis to visualize which segments of the time-series (e.g., pre-event noise or signal peaks) most influence the model’s decision.

🛠 Prerequisites
Required Libraries
Ensure your environment has the following installed:

Frameworks: TensorFlow 2.x, Keras, PyTorch

Optimization: keras-tuner

Data Science: numpy, pandas, scikit-learn, matplotlib, seaborn

Specialized: imbalanced-learn (for SMOTE), ObsPy (for seismic data processing)

Expected Data Format
The notebooks are configured to read metadata from pickle files:

tr_No_Fil.pkl (Training)

val_No_Fil.pkl (Validation)

te_No_Fil.pkl (Testing)

For the IM_based model, images should be organized in folders (e.g., train_IM_c123/) following the naming convention: {component}_{record_id}.png.

🚀 How to Use
Preprocessing: Use the data cleaning and restructuring functions (like restructure_dataframe_3) within the notebooks to format your seismic records.

Training: Run the Hyperparameter Tuning cells to find the optimal weights for your specific dataset.

Evaluation: The notebooks generate Confusion Matrices and Detailed Classification Reports (Precision, Recall, F1-Score) to assess performance on both internal and external datasets.

Explainability: Run the Occlusion Sensitivity cells in the ResNet notebook to generate heatmaps identifying critical regions of the waveforms.

📈 Performance Summary
ResNet: High internal accuracy (~93%), ideal for datasets consistent with the training distribution.

CNN (Image-based): Superior generalizability (75% accuracy on unseen external datasets), making it more robust for diverse seismic networks.

📝 Citation
If you use this code in your research, please cite:

Namin, A., Kottke, A., Thompson, E., & Esteghamati, M. Z. (2025). Generalizability and Explainability of Deep Learning Models to Assess Ground Motion Quality. Earthquake Spectra.
