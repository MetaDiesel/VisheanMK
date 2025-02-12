import sys
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
import boto3

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# S3 input and output paths
input_path = "s3://mkar/raw_reviews/"
output_path = "s3://mkar/processed_data/"
checkpoint_file = "s3://mkar/metadata/processed_files.txt"  # Stores processed file names

# Step 1: Get list of all raw files in the input folder
s3_client = boto3.client("s3")
bucket_name = "mkar"
prefix = "raw_reviews/"
processed_files_set = set()

# Read the checkpoint file (list of already processed files)
try:
    checkpoint_obj = s3_client.get_object(Bucket=bucket_name, Key="metadata/processed_files.txt")
    processed_files_set = set(checkpoint_obj["Body"].read().decode("utf-8").split("\n"))
except s3_client.exceptions.NoSuchKey:
    print("No checkpoint file found. Processing all raw data.")

# Get all current files in the raw folder
raw_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
new_files = [obj["Key"] for obj in raw_objects.get("Contents", []) if obj["Key"] not in processed_files_set]

# Step 2: If no new files, exit early
if not new_files:
    print("No new raw files found. Exiting...")
    sys.exit(0)

# Step 3: Read only new raw data
raw_dynamic_frame = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [f"s3://{bucket_name}/{file}" for file in new_files]},
    format="csv",
    format_options={"withHeader": True}
)

# Convert to DataFrame
raw_df = raw_dynamic_frame.toDF()

# Step 4: Keep only necessary columns and clean the data
expected_columns = ["Text", "Score"]
filtered_df = raw_df.select(*expected_columns)

# Step 5: Fill missing values and ensure Score is an integer
filtered_df = filtered_df.fillna({"Text": "", "Score": "1"})
filtered_df = filtered_df.withColumn("Score", F.col("Score").cast("int"))

# Step 6: Save the processed data
filtered_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# Step 7: Update the checkpoint file
new_processed_files = "\n".join(new_files)
s3_client.put_object(Bucket=bucket_name, Key="metadata/processed_files.txt", Body=new_processed_files)

print(f"Processing complete. Processed files saved to {output_path}")
