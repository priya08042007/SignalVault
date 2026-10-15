# 🛡️ SIGNAL VAULT — NETWORK INTELLIGENCE & PACKET ANALYTICS PLATFORM

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-0284c7.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59.2-ff4b4b.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Unit%20Tests-22%20Passing-10b981.svg)](tests/)
[![Security Engine](https://img.shields.io/badge/Security-Offline%20Analytical%20SOC-00e5ff.svg)](src/)

---

## 1. PROJECT OVERVIEW

**Signal Vault** is a college-level network intrusion and cybersecurity analytics platform designed for offline packet record inspection, statistical hypothesis testing, mathematical matrix factorization, evolutionary feature selection, and machine learning classification.

### Key Capabilities:
- **Offline Security Analytics:** Analyzes network flow records from the **UNSW-NB15** intrusion detection benchmark without live packet capturing, port sniffing, or network vulnerability exploitation.
- **Data Quality & Hygiene:** Automatic type inference, infinite/missing value imputation, duplicate reporting, and strict prevention of target data leakage.
- **Statistical Hypothesis Testing:** Independent Welch's t-test, Student's t-test, One-way ANOVA, and two-sample variance F-tests with plain-English interpretations and scientific guardrails.
- **Dimensionality Reduction:**
  - **PCA (Principal Component Analysis):** Orthogonal variance maximization, scree plots, cumulative explained variance, factor loadings, and 2D projections.
  - **SVD (Singular Value Decomposition):** Eckart-Young-Mirsky low-rank matrix approximation, singular value energy spectrum, and Frobenius reconstruction error tracking.
- **Genetic Algorithm (GA) Feature Selection:** Combinatorial heuristic search optimizing validation accuracy while penalizing redundant feature counts via a parsimony penalty.
- **Intrusion Detection Classifiers:** Leakage-safe Scikit-Learn pipelines (Random Forest, Decision Tree, Logistic Regression) evaluating Normal versus Attack traffic.
- **Three-Way Benchmark:** Direct comparison between Baseline (all original features), PCA (dense components), and GA (sparse feature subset) on identical validation splits.
- **Cyber SOC Dashboard:** Dark navy cybersecurity interface with Plotly charts and downloadable CSV/Markdown threat intelligence reports.

---

## 2. PROJECT ARCHITECTURE

```text
Honors_MiniProject/
├── app.py                      # Main Streamlit SOC Dashboard Orchestrator
├── requirements.txt            # Minimal, tested Python dependencies
├── README.md                   # Complete documentation & setup instructions
├── .gitignore                  # Git hygiene (ignoring large datasets & caches)
├── data/
│   ├── .gitkeep                # Git placeholder for data folder
│   └── UNSW_NB15_training-set.csv # (Target destination for downloaded CSV)
├── src/
│   ├── __init__.py             # Package descriptor
│   ├── data_loader.py          # Safe CSV loading, schema verification & setup guidance
│   ├── preprocessing.py        # Leakage guard, ColumnTransformer, cleaning & splitting
│   ├── eda.py                  # Univariate, bivariate, multivariate Plotly visualizations
│   ├── statistical_tests.py    # Welch's t-test, One-way ANOVA, Variance F-test
│   ├── pca_analysis.py         # Scree plot, cumulative variance, loadings, 2D scatter
│   ├── svd_analysis.py         # SVD factorization, low-rank approximation, Frobenius error
│   ├── genetic_algorithm.py    # Binary chromosome GA feature selection & convergence
│   ├── intrusion_model.py      # IntrusionDetectionPipeline (Baseline, PCA, GA models)
│   ├── evaluation.py           # Confusion matrices, ROC curves, comparative metrics
│   └── report_generator.py     # Markdown threat intelligence report & CSV exports
└── tests/
    ├── __init__.py
    ├── test_data_loader.py     # Tests for missing file, schema parsing, download notes
    ├── test_preprocessing.py   # Tests for leakage exclusion, inf replacement, split
    ├── test_statistical_tests.py # Tests for Welch t-test, ANOVA, F-test edge cases
    ├── test_dimensionality.py  # Tests for PCA variance and SVD low-rank error
    ├── test_genetic_algorithm.py # Tests for chromosome initialization & GA loop
    └── test_intrusion_model.py # Tests for baseline, PCA, GA pipelines & metrics
```

---

## 3. DATASET SETUP & DOWNLOAD INSTRUCTIONS

Signal Vault is configured to read from:
```text
data/UNSW_NB15_training-set.csv
```
Absolute Windows Path:
```text
c:\Users\Asus\OneDrive\Desktop\Honors_MiniProject\data\UNSW_NB15_training-set.csv
```

### Where to Obtain the Dataset:
1. **Kaggle (UNSW-NB15 Dataset by mrwellsdavid):**  
   [https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)  
   *Download only the single file:* `UNSW_NB15_training-set.csv` (~82,332 records, ~15 MB).
2. **UNSW Canberra Cyber Research Repository:**  
   [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)  
   Official research repository created by the Cyber Range Lab of UNSW Canberra.

### Setup Instructions:
1. Download `UNSW_NB15_training-set.csv`.
2. Move the file into the `data/` folder inside this workspace:
   ```powershell
   Copy-Item "C:\Users\Asus\Downloads\UNSW_NB15_training-set.csv" -Destination "c:\Users\Asus\OneDrive\Desktop\Honors_MiniProject\data\"
   ```
3. **Alternative (Direct Upload):** If you prefer not to copy files manually, you can launch the app without the dataset and upload the CSV using the **"Upload CSV via Dashboard"** sidebar option.

> [!NOTE]
> The application starts and runs cleanly even when the dataset has not yet been downloaded. It displays an interactive setup guide and architectural documentation until the data is loaded.

---

## 4. LOCAL INSTALLATION & EXECUTION (WINDOWS POWERSHELL)

### Prerequisites:
- Windows 10/11
- Python 3.11, 3.12, or 3.13 installed and added to `PATH`

### Step-by-Step PowerShell Commands:

1. **Open PowerShell in the Project Directory:**
   ```powershell
   cd c:\Users\Asus\OneDrive\Desktop\Honors_MiniProject
   ```

2. **Create a Python Virtual Environment:**
   ```powershell
   python -m venv .venv
   ```

3. **Activate the Virtual Environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   > [!IMPORTANT]
   > **Fixing PowerShell Script Execution Policy:**
   > If PowerShell displays an error stating `running scripts is disabled on this system`, run this safe command in your current PowerShell session to bypass the restriction:
   > ```powershell
   > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   > .\.venv\Scripts\Activate.ps1
   > ```
   > Alternatively, you can run all commands directly without script activation:
   > ```powershell
   > .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   > .\.venv\Scripts\python.exe -m streamlit run app.py
   > ```

4. **Upgrade Pip and Install Dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the Test Suite (Verify Everything Works):**
   ```powershell
   python -m unittest discover -s tests -v
   ```
   *(Expected output: 22 tests passing in under 1 second)*

6. **Launch Signal Vault:**
   ```powershell
   streamlit run app.py
   ```
   Your web browser will automatically open to `http://localhost:8501`.

---

## 5. DASHBOARD NAVIGATION GUIDE

The sidebar provides 10 integrated analytical modules:

| Section | Focus & Functionality |
| :--- | :--- |
| **1. Overview & System Status** | System health, dataset loading status, normal vs attack flow counts, attack percentage, memory footprint, and educational architecture overview when data is missing. |
| **2. Dataset Explorer** | Interactive data preview, data types, missing/infinite value counts, duplicate detection, and CSV export of preview records. |
| **3. Exploratory Data Analysis (EDA)** | Univariate histograms and box plots, categorical frequencies, bivariate scatter plots, grouped box plots, and multivariate correlation heatmaps. |
| **4. Statistical Hypothesis Testing** | Welch's t-test, Student's t-test, One-way ANOVA, and Variance F-test. Displays $H_0$, $H_1$, test statistics, p-values, decision rules, and plain-English interpretations. |
| **5. Principal Component Analysis (PCA)** | Standardized PCA with scree plot, cumulative variance curve, 80%/90%/95% threshold counters, factor loadings heatmap, and 2D scatter projection. |
| **6. Singular Value Decomposition (SVD)** | Truncated SVD matrix factorization ($X = U \Sigma V^T$), singular value spectrum, cumulative energy capture, and Frobenius low-rank reconstruction error heatmap. |
| **7. Genetic Algorithm Feature Selection** | Evolutionary heuristic search over feature masks with parsimony penalties, interactive convergence curve, and optimal feature mask extraction. |
| **8. Intrusion Detection Classifier** | Supervised classification (Random Forest, Decision Tree, Logistic Regression) targeting `label` (0 = Normal, 1 = Attack) with confusion matrix and ROC curves. |
| **9. Feature Strategy Comparison** | Direct benchmark between **Baseline** (all features), **PCA** (dense components), and **GA** (sparse features) on identical validation splits. |
| **10. Threat Reports & Exports** | Generate and download an executive Markdown threat intelligence report, descriptive statistics CSV, and dataset provenance JSON. |

---

## 6. MATHEMATICAL & THEORETICAL FOUNDATIONS

### A. Principal Component Analysis (PCA)
- **Objective:** Solves for orthogonal eigenvectors of the sample covariance matrix $\mathbf{C} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}$ to maximize projected variance.
- **Key Insight:** PCA produces **new synthetic components** (linear combinations of original features) ordered by eigenvalues $\lambda_i$. It compresses dimensionality but does **not** discard or directly select original network features.

### B. Singular Value Decomposition (SVD)
- **Objective:** Factorizes any real $m \times n$ matrix $\mathbf{X}$ into $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, where $\mathbf{U}$ and $\mathbf{V}$ are orthogonal matrices and $\mathbf{\Sigma} = \text{diag}(\sigma_1, \dots, \sigma_r)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
- **Eckart-Young-Mirsky Theorem:** Truncating $\mathbf{\Sigma}$ to rank $k$ yields the optimal low-rank matrix $\mathbf{X}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ that minimizes the Frobenius reconstruction error:
  $$\| \mathbf{X} - \mathbf{X}_k \|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

### C. Genetic Algorithm (GA) Feature Selection
- **Objective:** Searches the exponential $2^D$ combinatorial space of original input features.
- **Chromosome:** Binary mask $\mathbf{m} = [m_1, m_2, \dots, m_D] \in \{0, 1\}^D$.
- **Fitness Function with Parsimony Penalty:**
  $$\text{Fitness}(\mathbf{m}) = \text{Score}(\mathbf{m}) - \alpha \times \left(\frac{\sum_{i=1}^D m_i}{D}\right)$$
  where $\text{Score}$ is the validation balanced accuracy and $\alpha$ is a complexity penalty penalizing unnecessary features.

### D. Hypothesis Testing & Scientific Guardrails
- **Welch's t-test:** Used when testing $H_0: \mu_{\text{normal}} = \mu_{\text{attack}}$ because network bursts violate homoscedasticity ($\sigma^2_{\text{normal}} \neq \sigma^2_{\text{attack}}$).
- **One-way ANOVA:** Tests $H_0: \mu_1 = \mu_2 = \dots = \mu_k$ across attack categories.
- **Crucial Cautionary Principles:**
  1. A low p-value ($p < 0.05$) rejects the null hypothesis but **does not prove causation**.
  2. A p-value is **not a measure of effect size**; Cohen's $d$ is computed alongside the t-statistic to measure physical separation.

---

## 7. METHODOLOGICAL LIMITATIONS (COLLEGE EVALUATION NOTE)

1. **Single Training-Set CSV:** This project is designed around `UNSW_NB15_training-set.csv` to keep download size and resource overhead manageable on student laptops.
2. **Validation Partitioning:** Because only the training CSV is loaded, model evaluation relies on an internal stratified train/validation split (75/25 ratio). This internal validation split provides a sound proof-of-concept benchmark, but is distinct from the official, separate UNSW-NB15 testing partition.
3. **Data Leakage Safeguards:** All feature transformers (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`, `PCA`) are fitted strictly on the training partition and evaluated on the held-out validation partition.

---

## 8. TROUBLESHOOTING & FAQ

### Q1: The app displays "Dataset Pending: Setup Required"
- **Solution:** Place `UNSW_NB15_training-set.csv` inside the `data/` folder, or select **"Upload CSV via Dashboard"** in the sidebar and drag the file directly into the browser.

### Q2: Port 8501 is already in use
- **Solution:** Launch Streamlit on a custom port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```

### Q3: How do I speed up the Genetic Algorithm on an older laptop?
- **Solution:** In Section 7 (Genetic Algorithm), reduce **Population Size** to 10 and **Generations** to 4. Signal Vault automatically sub-samples the evaluation fold to 2,500 records so the algorithm converges in seconds.

---

## 9. LICENSE & ACADEMIC INTEGRITY

Developed for academic coursework, demonstrative research, and cybersecurity engineering defenses. Licensed under the MIT License.
