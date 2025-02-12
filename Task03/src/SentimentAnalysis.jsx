import { useState } from "react";

export default function SentimentAnalysis() {
  const [reviewText, setReviewText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    if (!reviewText) {
      alert("Please enter a review!");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(
        `https://wb1dpjh1hd.execute-api.ap-southeast-1.amazonaws.com/v1/predict?text=${encodeURIComponent(reviewText)}`
      );
      
      if (!response.ok) {
        throw new Error("Failed to fetch sentiment analysis");
      }
      
      const data = await response.json();
      setResult({
        sentiment: data.sentiment,
        confidence: (data.sentiment_score * 100).toFixed(2),
      });
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Error analyzing sentiment." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center p-8">
      <h2 className="text-2xl font-bold">Sentiment Analysis</h2>
      <p className="mb-4">Type a review below and click "Analyze":</p>
      <input
        type="text"
        value={reviewText}
        onChange={(e) => setReviewText(e.target.value)}
        placeholder="Enter your review..."
        className="w-80 p-2 border rounded-md"
      />
      <button
        onClick={analyzeSentiment}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>
      {result && (
        <div className="mt-4 text-lg">
          {result.error ? (
            <p className="text-red-500">{result.error}</p>
          ) : (
            <p>
              Sentiment: <strong>{result.sentiment}</strong> <br />
              Confidence: <strong>{result.confidence}%</strong>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
