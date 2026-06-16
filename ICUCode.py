from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer, MinMaxS>
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys

#Start Spark
spark = SparkSession.builder.appName("ICU_los_pipeline").getOrCreate()
print("Spark Session Initialized.")

# File path
CSV_PATH = "hdfs://master:9000/user/sat3812/mimic/raw/ICUSTAYS.csv" #HDFS path
HDFS_PATH = "hdfs:///user/sat3812/data/mimic"# HDFS base path
potential_columns = ["los", "los_hours", "los_days", "length_of_stay", "lengthofstay"] #possible LOS label columns
seed = 42 # for reproducibility
training_ratio = 0.8 # 80% training, 20% testing
columns_needed = {"subject_id", "hadm_id", "stay_id", "icustay_id"}

# Load uprocessed CSV data
df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)

# Standardize column names
for c in df.columns:
    df = df.withColumnRenamed(c, c.strip().lower()) # Remove spaces and lowercase
cols = set(df.columns)

# Search for columns we want to use
label = next((c for c in potential_columns if c in cols), None)  # Find matching LOS label column
if not label:
    sys.exit(f"Error: Could not find a LOS label column. Looked for: {potential_columns}") # Exit if not found
df = df.withColumn(label, F.col(label).cast("double")) # Ensure label is double for linear regression and random forest

#Data that is null or impssible value (negative)
df = df.filter(F.col(label).isNotNull() & (F.col(label) >= 0.0)) # Keep only valid LOS entries
df = df.withColumn("los_log", F.log1p(F.col(label))) 

# Empty lists for stages and feature columns
stages = [] 
feature_cols = []

# One Hot Encoding for Gender
if "gender" in cols:
    gender_idx = "gender_idx"
    gender_ohe = "gender_ohe"
    # Convert string to numeric index
    stages.append(StringIndexer(inputCol="gender", outputCol=gender_idx, handleInvalid="keep", stringOrderType="frequencyDesc"))
    # Convert index to one-hot encoded
    stages.append(OneHotEncoder(inputCol=gender_idx, outputCol=gender_ohe)) # Add to stage list
    feature_cols.append(gender_ohe) # Use the OHE result 


# Imputation and Scaling
# Exclude labels, identifiers, and not OHE gender
drop_cols = columns_needed.union({label, "los_log", "gender"}) # Columns to drop
numeric_cols = [c for c, dtype in df.dtypes if c not in drop_cols and dtype.split("(")[0] in ("double", "int", "bigint", "float", "decimal", "smallint", "tinyint")] # Identify numeric columns


#Imputation using Median
imputed_cols = [f"{c}__imp" for c in numeric_cols] 
imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=imputed_cols) 
stages.append(imputer) #Add to stage list
feature_cols.extend(imputed_cols) # Add imputed columns 

# Merge preprocessed data
merged_preprocessed = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip") # Merge all features
stages.append(merged_preprocessed) # Add to stage list

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler(inputCol="features_raw", outputCol="features") # Scale features to [0, 1]
stages.append(scaler) # Add to stage list

# Combine into preprocessed data pipeline
pipeline = Pipeline(stages=stages) # Create pipeline
preprocessed_df = pipeline.fit(df).transform(df) 

print(f"Preprocessing complete. Total features used: {len(feature_cols)}")

# Split the data after preprocessing
train, test = preprocessed_df.randomSplit([training_ratio, 1 - training_ratio], seed=seed) 

# MODEL 1: LINEAR REGRESSION
#train model 
lr = LinearRegression(featuresCol="features", labelCol=label, maxIter=10) 
lr_model = lr.fit(train) 

#Predict 
lr_pred = lr_model.transform(test)

#Evaluate
lr_rmse = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse").evaluate(lr_pred) #RMSE
lr_r2 = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="r2").evaluate(lr_pred) #R^2

#Reults
print(f"LR Test RMSE (LOS): {lr_rmse:.3f}") 
print(f"LR Test R^2 (LOS): {lr_r2:.3f}")
print("="*60)


#MODEL 2: RANDOM FOREST REGRESSION 
# Initialize and train the model
rf = RandomForestRegressor(featuresCol="features", labelCol=label, numTrees=50, seed=seed) 
rf_model = rf.fit(train)

# Predict 
rf_pred = rf_model.transform(test)

# Evaluate 
rmse_evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction") 
rf_rmse = rmse_evaluator.evaluate(rf_pred, {rmse_evaluator.metricName: "rmse"}) #RMSE
rf_r2 = rmse_evaluator.evaluate(rf_pred, {rmse_evaluator.metricName: "r2"}) #R^2

# Results
print(f"RF Test RMSE (Original LOS Scale): {rf_rmse:.3f}")
print(f"RF Test R^2 (Original LOS Scale): {rf_r2:.3f}")
print("="*60)

#Stop session
print("Complete. Stopping Spark Session.")
spark.stop()
