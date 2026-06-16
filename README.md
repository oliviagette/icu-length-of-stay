# Predicting ICU Length of Stay Using Spark-Based Regression
This project focuses on predicting Intensive Care Unit (ICU) length of stay (LOS) using distributed data processing and machine learning with Apache Spark. Accurate LOS prediction is important for optimizing hospital resource allocation and improving patient care.

By using the a subset of a publicly available ICU dataset obtained from Kaggle titled “HOSP & ICU Datasets (100,000 med-data from 2001–2019) datas clinical dataset and running PySpark across multiple virtual machines, this project shows the power of big data analytics and parallel computation in healthcare prediction tasks.

# Overview
Hospital intensive care units (ICUs) face constant pressure to manage resources effectively. Predicting patient length of stay helps clinicians and administrators plan staffing, bed allocation, and treatment scheduling.

This project implements a Spark-based regression pipeline to model and predict ICU LOS from patient demographics and clinical variables. Using distributed computing across one master node and three worker nodes, the pipeline handles large-scale data efficiently and performs both preprocessing and regression in a scalable manner.

# Key Features
## Data Preprocessing
- Implemented with PySpark for distributed processing
- Missing value handling (median imputation, row removal)
- Outlier detection and filtering
- One-hot encoding for categorical variables
- Scaling using StandardScaler and MinMaxScaler

## Model Approaches
Each team member implemented a different regression method for comparison:
- Linear Regression (Andrea) - Baseline model using standardized features
- Random Forest Regression (Olivia) - Nonlinear ensemble model using scaled features
Both models were trained and evaluated on an 80/20 train-test split using the same preprocessed dataset.

# Results
## Compare Models
| Model | RMSE (days) | R^2 | Description |
|---|---|---|---|
| Linear Regression | 9.778 | 0.017 | Baseline model |
| Random Forest | 9.567 | 0.059 | Slight improvement, captures some nonlinearities |

Both models show limited predictive power, indicating that LOS might depend on additional variables or nonlinear relationships. Future improvements might include log-transformed targets, feature expansion, and hyperparameter tuning with CrossValidator.

## Performance Comparison
| Number of VMs | Cores Used (8 per VM) | Duration (mins) | Duration (seconds) |
|---|---|---|---|
| 1 | 8 | 1.7 | 102.0 |
| 2 | 16 | 1.4 | 84.0 |
| 3 | 24 | 1.4 | 84.0 |
| 4 | 32 | 1.5 | 90 |

The runtime decreased as the number of VMs increased from one to three, showing that Spark’s distributed processing improved efficiency by about 20%. However, performance slightly declined with four VMs, likely due to increased communication overhead between nodes.

# Distributed Cluster Setup
To handle the dataset’s scale, the project was deployed on four virtual machines (VMs):
- 1 Master Node and 3 Worker Nodes
- Configured with passwordless SSH and Spark Standalone mode
- Shared Spark installation at /opt/spark
- HDFS used for data storage and retrieval
Cluster configuration files (spark-env.sh, workers) ensured consistent runtime environments and parallel task execution.

# Dataset
We had originally planned on using the MIMIC-IV Version 3.1 dataset from PhysioNet, but had troubles retrieving the data due to permission issues. We ended up using a dataset we found on Kaggle that gave us practically the same data.

Source: [HOSP&ICU-datasets(100000 med-data from 2001-2019）]([url](https://www.kaggle.com/datasets/luciadam/icu-datasets?resource=download))

Reference: Kaggle. (2024). HOSP & ICU Datasets (100,000 med-data from 2001–2019) [Dataset]. Retrieved from https://www.kaggle.com/datasets/luciadam/icu-datasets. 

# Methods Summary
## Preprocessing
1. Read CSV data from HDFS
2. Handle missing or invalid LOS/age values
3. Impute missing numeric fields with the median
4. One-hot encode categorical variables
5. Assemble and scale features with VectorAssembler and MinMaxScaler
## Modeling
- Train regression models on Spark MLlib
- Evaluate using Root Mean Square Error and R^2
- Compare distributed performance with 1-4 VMs

# How to Run Project
## Prereqs
- Python 3.x
- Apache Spark installed on all VMs (/opt/spark)
- PySpark, NumPy, Pandas, Matplotlib, Seaborn

# Team Members
- Andrea Wroblewski
- Olivia Gette
