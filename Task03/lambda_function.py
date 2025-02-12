import json
import joblib
import boto3
import os

# AWS S3 Setup
s3 = boto3.client("s3")
bucket_name = "your-bucket-name"
model_path = "/tmp/sentiment_model.pkl"
vectorizer_path = "/tmp/vectoriser.pkl"

# Load model from S3 into Lambda
def load_model():
    if not os.path.exists(model_path):
        s3.download_file(bucket_name, "models/sentiment_model.pkl", model_path)
        s3.download_file(bucket_name, "models/vectoriser.pkl", vectorizer_path)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

MODEL, VECTORIZER = load_model()  # Load once on cold start

# Lambda Handler Function
def lambda_handler(event, context):
    try:
        # Parse incoming request
        body = json.loads(event["body"])
        text = body.get("text", "")

        if not text:
            return {"statusCode": 400, "body": json.dumps({"error": "No text provided"})}

        # Preprocess & predict
        text_vectorized = VECTORIZER.transform([text])
        prediction = MODEL.predict(text_vectorized)[0]

        # Return response
        response = {"prediction": str(prediction)}
        return {"statusCode": 200, "body": json.dumps(response)}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
