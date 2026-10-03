# Household Energy Pattern Segmentation and Consumption Prediction Using Machine Learning

> **IBM SkillsBuild Data Analytics with AI Internship Project**

---

## Problem Statement

Household appliance energy consumption varies across different times of the day, days of the week, and seasons of the year. Understanding these energy-consumption patterns can help identify periods of relatively higher and lower appliance usage.

This project analyzes time-stamped household appliance energy-consumption observations using machine learning. K-Means Clustering is used to identify recurring energy-consumption patterns, while supervised regression models are used to predict appliance energy consumption for the following 10-minute interval.

The project provides a structured and reproducible example of applying data preprocessing, exploratory data analysis, feature engineering, unsupervised learning, supervised learning, dimensionality reduction, and model evaluation to a real-world energy dataset.

---

## Objectives

1. Load and explore a real-world household energy-consumption dataset.
2. Perform data-quality checks and preprocessing.
3. Conduct Exploratory Data Analysis (EDA) to understand energy-consumption distributions and temporal patterns.
4. Engineer meaningful time-based, rolling, and lag features.
5. Apply K-Means Clustering to identify recurring energy-consumption patterns.
6. Evaluate different cluster counts using the Elbow Method and Silhouette Score.
7. Visualize the resulting cluster structure using Principal Component Analysis (PCA).
8. Build supervised regression models to predict the next 10-minute appliance energy consumption.
9. Evaluate regression models using MAE, RMSE, and R².
10. Interpret the identified energy-consumption patterns and generate data-driven observations.

---

## Technologies

| Tool / Library | Purpose |
|---|---|
| Python | Programming language |
| pandas | Data loading, cleaning, transformation, and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning, clustering, scaling, PCA, and evaluation |
| Jupyter Notebook / Google Colab | Interactive development environment |

---

## Dataset

This project uses the **UCI Appliances Energy Prediction** dataset.

The dataset contains time-stamped measurements of appliance energy consumption together with indoor environmental measurements and outdoor weather-related variables.

### Dataset Information

- **Dataset:** Appliances Energy Prediction
- **Source:** UCI Machine Learning Repository
- **Target variable:** `Appliances`
- **Time column:** `date`
- **Observations:** 19,735
- **Sampling interval:** 10 minutes
- **Dataset type:** Multivariate time-series data

The dataset contains variables related to:

- Appliance energy consumption
- Lighting energy consumption
- Indoor temperature
- Indoor relative humidity
- Outdoor temperature
- Outdoor relative humidity
- Atmospheric pressure
- Wind speed
- Visibility
- Dew point temperature

### Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

The dataset file used by the project is:

```text
energydata_complete.csv
```

> The dataset itself is not included in this GitHub repository. Download it from the source above before running the notebook.

---

## Dataset Configuration

The actual dataset uses one combined datetime column and the appliance-energy target:

```python
FILE_PATH = "energydata_complete.csv"
DATE_COLUMN = "date"
TIME_COLUMN = None
TARGET_COLUMN = "Appliances"
```

For Google Colab, the completed project used the dataset from Google Drive.

---

## Methodology

The project follows an end-to-end machine-learning workflow:

```text
1. Data Loading
        ↓
2. Data Quality Check
        ↓
3. Data Cleaning & Preprocessing
   - duplicate removal
   - target validation
   - missing-value handling
   - datetime conversion
   - sorting
   - IQR-based target outlier filtering
        ↓
4. Exploratory Data Analysis
   - consumption distribution
   - time-based patterns
   - correlation analysis
        ↓
5. Time-Based Feature Engineering
   - Hour
   - DayOfWeek
   - Month
   - Year
   - IsWeekend
   - Quarter
   - TimeOfDay
   - Season
        ↓
6. Rolling & Lag Features
   - Rolling_Mean_24
   - Rolling_Std_24
   - Lag_1
   - Lag_24
        ↓
7. K-Means Clustering
   - feature selection
   - StandardScaler
   - K-Means
        ↓
8. Cluster Evaluation
   - Elbow Method
   - Silhouette Score
        ↓
9. PCA Visualization
        ↓
10. Supervised Prediction
    - next 10-minute target
    - chronological train-test split
        ↓
11. Regression Models
    - Linear Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
        ↓
12. Model Evaluation
    - MAE
    - RMSE
    - R²
        ↓
13. Cluster Profiling & Interpretation
        ↓
14. Energy-Use Insights
        ↓
15. Conclusion
```

---

## Clustering

K-Means Clustering is used to identify recurring energy-consumption states based on selected consumption, temporal, and engineered features.

The Elbow Method and Silhouette Score are considered together when evaluating different values of `k`.

A final configuration of:

```text
k = 10
```

was selected as a suitable configuration for the project based on the combination of cluster-evaluation results and interpretability.

The clustering stage should be interpreted as segmentation of **time-based energy-consumption observations**, rather than as identification of ten physically separate households.

---

## PCA Visualization

Principal Component Analysis (PCA) is used to reduce the clustering feature space to two principal components for visualization.

The first two principal components retain approximately:

```text
54.91% of the variance
```

This provides a two-dimensional representation of the cluster structure.

---

## Prediction Task

The supervised-learning stage predicts appliance energy consumption for the **following 10-minute interval**.

The target is created using the next observation:

```python
df["Target_Next_10min"] = df["Appliances"].shift(-1)
```

A chronological train-test split is used rather than a random split so that future observations are not mixed into the training data.

### Prediction Features

The prediction model uses temporal, environmental, and lag-based features, including:

- Hour
- DayOfWeek
- Month
- IsWeekend
- Quarter
- TimeOfDay
- Season
- Lighting consumption
- Indoor temperature and humidity variables
- Outdoor weather variables
- Lag_1
- Lag_24

Target-derived rolling statistics and the cluster label are excluded from the prediction inputs to reduce direct target leakage.

---

## Regression Models

Four regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

---

## Model Evaluation

The models were evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted consumption. Lower values indicate smaller average prediction errors.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors. Lower values indicate better predictive performance.

### R² Score

Measures the proportion of variance in the target that is explained by the model. Higher values indicate better explanatory performance on the evaluated test data.

### Final Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 12.1841 | 17.8987 | 0.5473 |
| Decision Tree | 16.6273 | 25.4224 | 0.0868 |
| Random Forest | 16.2539 | 21.6255 | 0.3392 |
| **Gradient Boosting** | **11.4610** | **16.9651** | **0.5933** |

Among the models tested in this experiment, **Gradient Boosting produced the lowest MAE and RMSE and the highest R²** on the held-out test period.

---

## Cluster Analysis

The final K-Means model produced 10 energy-consumption patterns.

Some clusters represented relatively higher average appliance consumption, while others represented lower-consumption states. The patterns also differed according to temporal characteristics such as:

- Time of day
- Weekday vs. weekend
- Month
- Season
- Consumption variability

### Cluster Observations

| Cluster | Average Appliances (Wh) | Pattern |
|---|---:|---|
| 0 | 88.85 | Higher Evening Weekend Winter Pattern |
| 1 | 55.63 | Moderate Night Weekday Spring Pattern |
| 2 | 44.77 | Lower Afternoon Weekday Winter Pattern |
| 3 | 62.35 | Moderate Afternoon Weekday Spring Pattern |
| 4 | 46.18 | Lower Night Weekend Winter Pattern |
| 5 | 48.34 | Lower Night Weekday Winter Pattern |
| 6 | 88.53 | Higher Evening Weekend Spring Pattern |
| 7 | 91.66 | Higher Evening Weekday Spring Pattern |
| 8 | 53.86 | Lower Night Weekend Spring Pattern |
| 9 | 87.47 | Higher Evening Weekday Winter Pattern |

The highest average appliance consumption among the clusters was observed in **Cluster 7**, while the lowest average was observed in **Cluster 2**.

These values describe energy-consumption observations assigned to each cluster and should not be interpreted as measurements from ten separate physical households.

---

## Key Insights

The analysis produced several data-driven observations:

- Appliance energy consumption varies substantially across time.
- Higher-consumption patterns are associated with particular evening periods in the clustered observations.
- Lower-consumption patterns occur during several night and afternoon periods.
- Weekday and weekend observations form different recurring patterns.
- Seasonal context contributes to differences between some consumption states.
- Lagged consumption information is useful for predicting the following 10-minute appliance consumption.
- Gradient Boosting provided the strongest predictive performance among the four tested regression models.

---

## Practical Interpretation

The identified patterns can support analytical tasks such as:

- Understanding periods of relatively high appliance usage.
- Comparing weekday and weekend consumption behavior.
- Studying changes in consumption across seasons.
- Supporting future energy-demand monitoring systems.
- Identifying time periods that may deserve further investigation for energy-efficiency analysis.

These observations are analytical findings from the dataset and do not by themselves establish actual energy savings or household behavior beyond the available data.

---

## Project Outputs

| Output | Description |
|---|---|
| Data quality summary | Dataset structure, missing values, duplicates, and cleaning information |
| EDA visualizations | Energy-consumption and temporal analysis |
| Time-based analysis | Hourly, weekday/weekend, and seasonal patterns |
| Elbow curve | K-Means inertia across different cluster counts |
| Silhouette analysis | Cluster-quality comparison |
| PCA visualization | Two-dimensional cluster representation |
| Cluster profiles | Mean characteristics of each energy pattern |
| Model comparison | MAE, RMSE, and R² for four regression models |
| Prediction evaluation | Comparison of actual and predicted consumption |
| Energy insights | Interpretation of the identified consumption patterns |

---

## Project Structure

```text
Household-Energy-Pattern-Segmentation/
│
├── Household_Energy_Pattern_Segmentation_and_Prediction.ipynb
├── README.md
├── requirements.txt
├── PROJECT_REPORT_OUTLINE.md
├── PROJECT_REPORT.docx
│
└── data/
    └── README.md
```

The dataset is not included in the repository because it is downloaded separately from the UCI Machine Learning Repository.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the **UCI Appliances Energy Prediction** dataset from:

https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

Place:

```text
energydata_complete.csv
```

in the appropriate project/data location.

### 5. Run the Notebook

The notebook can be opened using Jupyter Notebook or Google Colab.

```bash
jupyter notebook
```

Then open:

```text
Household_Energy_Pattern_Segmentation_and_Prediction.ipynb
```

---

## Reproducibility

The project uses fixed random states where applicable so that the clustering and machine-learning workflow can be reproduced under the same environment and dataset conditions.

The analysis is designed for educational and internship-project purposes and can be extended with additional datasets, models, and time-series forecasting techniques.

---

## Limitations

- The dataset represents a specific measurement period and environment.
- The analysis does not represent all households or all geographic regions.
- K-Means assumes that observations can be meaningfully grouped using distance-based similarity.
- The prediction task estimates the next 10-minute observation rather than performing long-term energy forecasting.
- Model performance depends on the available features and the characteristics of the dataset.
- The clustering represents recurring energy-consumption states rather than separately identified physical households.

---

## Future Work

Possible extensions include:

- Testing additional clustering algorithms such as DBSCAN or hierarchical clustering.
- Exploring advanced time-series forecasting methods.
- Testing additional regression algorithms.
- Performing systematic hyperparameter tuning.
- Incorporating longer observation periods or additional household datasets.
- Developing an interactive dashboard for energy-pattern exploration.
- Adding real-time prediction and monitoring capabilities.

---

## Conclusion

This project demonstrates an end-to-end machine-learning workflow for analyzing household appliance energy consumption.

K-Means Clustering was used to segment recurring energy-consumption patterns, while PCA provided a compact visualization of the resulting cluster structure. Four regression models were evaluated for next-10-minute appliance consumption prediction, with Gradient Boosting achieving the strongest performance among the tested models.

Overall, the project demonstrates how data preprocessing, feature engineering, unsupervised learning, supervised learning, and model evaluation can be combined to extract meaningful analytical information from real-world energy data.

---

## References

1. UCI Machine Learning Repository — Appliances Energy Prediction Dataset  
   https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

3. McKinney, W. — *Python for Data Analysis*.

4. IBM SkillsBuild — Data Analytics with AI Internship learning resources.

---

## License

This project is developed for educational purposes as part of the **IBM SkillsBuild Data Analytics with AI Internship** program.
