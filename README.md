# CFBoost Reproduction: Combating Heterogeneous Model Biases

This repository contains the reproduction code for the WSDM '25 paper: **"Combating Heterogeneous Model Biases in Recommendations via Boosting"**.

This project is a fork of the [original repository](https://github.com/JP-25/CFBoost). It includes significant code refactoring, bug fixes for missing data attributes, and custom scripts to reproduce the tables and figures presented in the paper.

## 🚀 Key Improvements in This Fork
This version provides:

1.  **Data Generation Scripts:** Automates the creation of missing `.npy` attributes (`user_mainstream.npy`, `item_popularity.npy`, etc.) which were absent in the original repo but are required for bias evaluation.
2.  **Code Refactoring:** 
    *   Separated **Design 1 (CFAdaBoost)** and **Design 2 (CFBoost)** into distinct, standalone model files for easier execution and debugging.
    *   Modified the base models to calculate **MDG (Mean Discounted Gain)** for item-side fairness evaluation, which was not implemented in the original code.
3.  **Reproduction Scripts:** One-click Python scripts to generate Tables 1-4 and Figures 1, 3, and 4 as high-quality images.

---

## 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AgentBubu/CFBoost-Repro.git
    cd CFBoost-Repro
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch numpy pandas scipy matplotlib tqdm
    ```

---

## 📊 Step 1: Data Preparation

The original repository assumes certain pre-calculated attribute files exist. This repo generates them for you based on the dataset.

1.  Ensure your dataset (e.g., `amazon_cds`) is located in `data/recsys_data/`.
2.  Run the generation script:
    ```bash
    python generate_missing_files.py
    ```
    *This creates `user_activeness.npy`, `user_mainstream.npy`, `item_popularity.npy`, and `item_mainstream.npy`.*

---

## 🧠 Step 2: Training the Models

To reproduce the results, you need to train three specific models. Configuration files (`.cfg`) have been pre-set for the **Amazon CDs** dataset.

### 1. Matrix Factorization (Baseline)
This runs the standard MF model, modified to calculate item-side metrics.
```bash
python main.py --model_name MF
```

## 📈 Step 3: Reproducing Tables & Figures
Once training is complete, you can generate the exact visualizations from the paper using these custom scripts.
```bash
python reproduce_tables_img.py
```
```bash
python reproduce_figure_1.py
```
```bash
python reproduce_figure_3.py
```
```bash
python reproduce_figure_4.py
```

---
## 📝 Citation
If you use this code, please cite the original paper:
@inproceedings{pan2025combating,
  title={Combating Heterogeneous Model Biases in Recommendations via Boosting},
  author={Pan, Jinhao and Caverlee, James and Zhu, Ziwei},
  booktitle={WSDM '25},
  year={2025}
}
