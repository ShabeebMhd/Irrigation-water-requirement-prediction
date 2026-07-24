## Technical Workflow & Core Pipeline Architecture

The Sri Lanka Irrigation Predictor implements a robust, end-to-end Machine Learning system optimized for heavy zero-inflation and group-dependent agronomic records. The implementation enforces zero-leakage constraints across all processing stages.



Raw Data (10k Rows) ──> GroupShuffleSplit (80/20 on Field_Cycle_ID)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   Stage 1: Classification               Stage 2: Regression
   [Random Forest Classifier]           [Gradient Boosting Regressor]
   - Determines: Needs Water? (Y/N)     - Determines: Target Volume (mm)
   - Benchmarked ROC AUC: 0.9946        - Trained ONLY on positive rows (y > 0)
   - Features: 9 Optimal Fields         - Features: 5 Key Agronomic Drivers
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                Blended Hurdle Pipeline Output




### 1. Data Ingestion, Cleaning & Leakage Prevention
* **High-Correlation Pruning**: Dropped `Gross_Irrigation_Requirement_mm` (0.976 correlation with target) and `Previous_Irrigation_mm` to prevent direct target leakage.
* **Dimensionality Reduction**: Dropped `District` to eliminate structural noise, as its spatial signals are perfectly captured by `Climate_Zone`.
* **Group-Safe Splitting**: Utilized a `GroupShuffleSplit` (80/20 train/test split) explicitly grouped on `Field_Cycle_ID`. This guarantees that rows belonging to the same crop growth stages are never leaked between the training set and testing evaluation set.
* **Row Identifier Exclusion**: Automatically removed `Field_Cycle_ID` from the active feature space ($X$) prior to model fitting to prevent the model from assigning false predictive importance to arbitrary row indices.

### 2. Feature Engineering & Preprocessing Pipeline
* **Categorical Encoding**: Multi-class columns (`Season`, `Climate_Zone`) are passed through a deterministic `OneHotEncoder(handle_unknown='ignore')` fit strictly on the training partition.
* **Feature Scaling**: Implemented independent `StandardScaler` instances for each modeling phase. The classification scaler transforms the entire feature matrix, while the regression scaler is fit strictly on positive-irrigation observations to preserve the true variance of water volumes.
* **Outlier Preservation**: Outliers in highly skewed climatic features (`Rainfall_mm`, `Stage_Duration_days`) were deliberately preserved in the training dataset to ensure the downstream ensemble models maintain high generalization performance during erratic monsoon spikes.

### 3. Two-Stage Hurdle Architecture
Standard single-stage regressors break down on this dataset due to a massive **57.02% zero-inflation ceiling**. To overcome this, the architecture splits the task into two highly specialized models:

#### Stage 1: The Classification Hurdle (Gatekeeper)
* **Objective**: Binary classification to predict if a field requires any irrigation at all ($Net\_Irrigation\_Requirement\_mm > 0$).
* **Feature Selection**: Recursive Feature Elimination with Cross-Validation (`RFECV`) paired with `GroupKFold(5)` isolated **9 optimal features** (including `Season`, `Rainfall_mm`, `Reference_ET0_mm_day`, and `Crop_Coefficient_Kc`).
* **Model & Performance**: A tuned **Random Forest Classifier** with balanced class weights.
  * **Accuracy**: 96.0%
  * **ROC AUC**: 0.9946
  * **F1-Score**: 0.9533

#### Stage 2: The Volume Regressor (Quantity Model)
* **Objective**: Continuous regression to predict the exact volume of water needed in millimeters ($mm$). Trained exclusively on the subset of rows where irrigation is greater than zero.
* **Feature Selection**: Feature pruning via Random Forest feature importance narrowed the feature space down to the **5 most critical physical drivers**: `Stage_Duration_days`, `Rainfall_mm`, `Reference_ET0_mm_day`, `Crop_Coefficient_Kc`, and `Tmax_C`.
* **Model & Performance**: A hyperparameter-tuned **Gradient Boosting Regressor** that comfortably outperforms standard Random Forest baselines across all validation splits:
  * **Cross-Validation MAE**: 9.69mm
  * **Test Set MAE**: 9.47mm
  * **Test $R^2$ Score**: 0.9203

---

### 4. Consolidated Production Performance
When evaluated end-to-end on the complete 2,000-row test dataset (including all true zero-irrigation records), the Two-Stage Hurdle Pipeline delivers superior performance compared to both traditional physics formulas and single-stage ML baselines:

| Evaluation Metric | FAO-56 Physics Formula | Single-Stage Random Forest | Two-Stage Hurdle Pipeline (Ours) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 6.74 mm | 4.45 mm | **4.30 mm** |
| **Root Mean Squared Error (RMSE)** | 13.39 mm | 9.46 mm | **9.24 mm** |
| **Coefficient of Determination ($R^2$)** | 0.8953 | 0.9478 | **0.9502** |

* **Hurdle Threshold Justification**: The pipeline uses an operationally calibrated threshold of `0.50`. This specific value directly minimizes raw global MAE while balancing the critical real-world business tradeoff between water wastage (False Positives) and crop stress from missed irrigation (False Negatives).

---

### 5. Model Trust, Explainability & Edge-Case Stress Testing
* **SHAP Interpretability**: Integrated a `shap.TreeExplainer` on both pipeline components. The generated beeswarm and summary feature importance plots confirm strict alignment with FAO-56 thermodynamic physics: high `Rainfall_mm` drives predictions strongly toward zero, while high `Reference_ET0_mm_day` (evapotranspiration) serves as the primary upward volume driver.
* **Drought Edge-Case Resiliency**: The pipeline was stress-tested against severe drought conditions (defined as the top 25% highest evapotranspiration values combined with the bottom 10% lowest rainfall records). 
  * The classifier successfully flagged **100%** of severe drought rows as needing water (Mean probability: `0.982`).
  * While raw MAE scales up to `10.21mm` due to the naturally high volume demands of drought cycles (averaging 74.91mm), the model's relative error actually **improves to 13.6%** (compared to 18.2% on the standard test set), proving extreme reliability under climate stress.
