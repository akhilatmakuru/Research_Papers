# Transfer Learning + Autoencoder Pipeline for MRI-Based Cognitive Prediction

This repository contains the implementation of the **transfer learning pipeline**, **autoencoder architecture**, and **regression + cognitive stage classification models** developed as part of the thesis:

> *“Explainable and Data-Efficient Machine Learning for MRI-Based Alzheimer’s Disease Prediction.”*

The following code enables full reproducibility of the experiments, including dataset preprocessing, feature scaling, deep learning model training, and transfer-learning-based cognitive staging.

---

## 📁 Repository Structure

```
/root
│
├── transfer_learning_autoencoder.py     # Main pipeline
├── requirements.txt                     # Environment dependencies
├── README.md                            # Reproducibility documentation
└── /data                                # (optional) placeholder for user dataset
```

---

# 1. 🔧 Environment Setup

### **1.1. Create and activate a virtual environment**

```bash
python3 -m venv mri_env
source mri_env/bin/activate
```

### **1.2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **Key Libraries Used**

* TensorFlow / Keras
* Scikit-Learn
* Imbalanced-Learn
* Pandas, NumPy
* Matplotlib, Seaborn

Your repo’s `requirements.txt` should already list all pinned versions.

---

# 2. 📊 Dataset Requirements

The code expects the MRI dataset:

```
MRI_rois_alldiseases.csv
```

> *The dataset used in this thesis (MRI ROI data) is licensed and cannot be redistributed.
> Users must obtain access independently through ADNI/AIBL/IXI and place the CSV file in the following directory:*
>
> ```
> /data/MRI_rois_alldiseases.csv
> ```

---

# 3. ▶️ How to Run the Pipeline

The entire workflow is contained in the script:

```
transfer_learning_autoencoder.py
```

Run it with:

```bash
python transfer_learning_autoencoder.py
```

This will automatically execute:

1. **Data Loading**
2. **Filtering for MCI, EMCI, LMCI, Mild, Moderate AD**
3. **Label encoding + resampling**
4. **Train–test split**
5. **Standard scaling**
6. **Initial regression model (baseline)**
7. **Transfer learning via encoder extraction**
8. **Autoencoder training**
9. **Final regression + cognitive stage classification**
10. **Performance reporting (MAE, RMSE, MSE, Accuracy)**

---

# 4. 🧠 Methodology Breakdown

Below is a simple description of how each block of your code works — perfect for examiners checking reproducibility.

---

## **4.1. Baseline Regression Model**

* Input size: 401 MRI ROI features
* Final output: 1 neuron (continuous MMSE prediction)
* Loss: MSE
* Optimizer: RMSProp
* Early stopping enabled

**Outputs:**

* MAE on test set
* Predicted MMSE values

---

## **4.2. Transfer Learning Step**

The encoder from the baseline network is extracted:

```python
encoder_model = Model(inputs=model.input, outputs=model.layers[-8].output)
```

* Encoder layers are **frozen**
* Encoder embeddings are used as input for the autoencoder

---

## **4.3. Autoencoder Architecture**

* Input: 401 features
* Encoder architecture
* Transfer-learned layer concatenated with encoder
* Decoder reconstructs original input
* Loss: MSE
* Optimizer: Adam or RMSprop
* Trained for up to 1000 epochs with early stopping

**Output:**

* Trained autoencoder
* Learned 60-dim embeddings

---

## **4.4. Final Regression + Categorization Model**

For 10 iterations:

1. Use encoder to extract embeddings
2. Train regression head
3. Predict MMSE
4. Convert MMSE predictions → cognitive stages

   * 15–20 → Mild
   * 21–30 → Moderate
5. Compute metrics:

   * MAE
   * RMSE
   * MSE
   * Accuracy
   * Confusion matrix

At the end, mean + SD of metrics are printed.

---

# 5. 📤 Outputs Generated

The script prints:

### **Regression Performance**

* Average MAE
* Standard deviation of MAE
* Average RMSE
* Average MSE

### **Classification Performance**

* Average accuracy
* Standard deviation of accuracy
* Confusion matrices (per iteration)

