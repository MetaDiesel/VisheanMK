import boto3
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# S3 bucket details
bucket_name = "mkar"
processed_data_prefix = "processed_data/"
local_file = "/tmp/processed_reviews.csv"  # Glue uses /tmp for local storage

# Step 1: Find the latest processed file in S3
def get_latest_processed_file():
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=processed_data_prefix)

    if "Contents" not in response:
        raise FileNotFoundError("No processed data files found in S3.")

    # Extract file names and sort by last modified date
    files = response["Contents"]
    latest_file = max(files, key=lambda x: x["LastModified"])["Key"]

    print(f"Latest processed file found: {latest_file}")
    return latest_file

# Step 2: Download the latest processed file from S3
def download_data_from_s3():
    latest_file_key = get_latest_processed_file()
    
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, latest_file_key, local_file)

    print(f"Downloaded {latest_file_key} from S3 to {local_file}")

# Step 3: Load and preprocess the data
def preprocess_data():
    # Read CSV safely
    df = pd.read_csv(local_file, on_bad_lines="skip", encoding="utf-8")

    # Ensure only necessary columns exist
    expected_columns = ["Text", "Score"]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Missing required columns: {expected_columns}. Found columns: {df.columns.tolist()}")

    df = df[expected_columns]

    # Drop rows with missing values
    df = df.dropna()

    # ✅ NEW: Keep only rows where "Score" is a valid integer
    df = df[df["Score"].astype(str).str.isnumeric()]  # Keep only numeric Score values
    df["Score"] = df["Score"].astype(int)  # Convert to int

    X = df["Text"]
    y = df["Score"]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# Step 4: Train a Logistic Regression model
def train_model(X_train_tfidf, y_train):
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

# Step 5: Save model and upload to S3
def save_model_and_vectorizer(model, vectorizer):
    model_path = "/tmp/sentiment_model.pkl"
    vectorizer_path = "/tmp/vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    s3 = boto3.client("s3")
    s3.upload_file(model_path, bucket_name, "models/sentiment_model.pkl")
    s3.upload_file(vectorizer_path, bucket_name, "models/vectorizer.pkl")

    print("Model and vectorizer saved and uploaded to S3")

# Step 6: Run the training pipeline
def main():
    download_data_from_s3()
    
    # Preprocess data safely
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data()

    # Train model
    model = train_model(X_train_tfidf, y_train)

    # Save & upload model
    save_model_and_vectorizer(model, vectorizer)

    print("Training completed and model uploaded to S3.")

# Execute the training job
if __name__ == "__main__":
    main()
