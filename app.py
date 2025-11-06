import nltk
from flask import Flask, request, jsonify
from textblob import TextBlob
import re

# Download resources needed by TextBlob/NLTK (only needs to be run once)
try:
    nltk.download('punkt', quiet=True)
except:
    print("NLTK Punkt not found, attempting to download.")
    
app = Flask(__name__)

# --- NLP Preprocessing Function ---
def preprocess_review(text):
    """Clean the text before sentiment analysis."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation (keep spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and rejoin (TextBlob handles some of this internally, but good practice)
    # This minimal cleaning is often enough for simple TextBlob sentiment
    return text

# --- Sentiment Analysis Function ---
def analyze_sentiment(review_text):
    """Performs sentiment analysis using TextBlob."""
    cleaned_text = preprocess_review(review_text)
    
    # Create a TextBlob object
    analysis = TextBlob(cleaned_text)
    
    # Get the sentiment properties
    # polarity: a float in the range [-1.0, 1.0] where 1.0 is positive sentiment and -1.0 is negative sentiment.
    # subjectivity: a float in the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    polarity = analysis.sentiment.polarity
    
    # Determine the sentiment label
    if polarity > 0.1:
        sentiment_label = "POSITIVE"
    elif polarity < -0.1:
        sentiment_label = "NEGATIVE"
    else:
        sentiment_label = "NEUTRAL"
        
    return {
        "text": review_text,
        "sentiment_label": sentiment_label,
        "polarity_score": round(polarity, 4)
    }

# --- API Endpoint ---
@app.route('/api/analyze_review', methods=['POST'])
def api_analyze_review():
    # Expects JSON data in the request body
    data = request.get_json()
    if not data or 'review_text' not in data:
        return jsonify({"error": "Missing 'review_text' in request body"}), 400
    
    review_text = data['review_text']
    
    try:
        # Get the analysis result
        result = analyze_sentiment(review_text)
        return jsonify(result), 200
    except Exception as e:
        # General error handling
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

# --- Main Run Block ---
if __name__ == '__main__':
    # Use '0.0.0.0' for external access, if needed
    print("Starting Sentiment Analysis API on http://127.0.0.1:5000/api/analyze_review")
    app.run(debug=True)