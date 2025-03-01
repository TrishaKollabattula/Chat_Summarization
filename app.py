from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load a lighter abstractive summarization model
abstractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

# Function for extractive summarization
def extractive_summarization(text, num_sentences=3):
    sentences = text.split('. ')
    if len(sentences) < num_sentences:
        return text

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]
    summary = '. '.join(ranked_sentences[:num_sentences])
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    summary_type = data.get('type', 'abstractive')

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Limit input to 1000 characters to manage memory
    if len(text) > 1000:
        return jsonify({"error": "Text too long, max 1000 characters"}), 400

    try:
        if summary_type == 'abstractive':
            summary = abstractive_summarizer(text, max_length=100, min_length=30, do_sample=False)
            summary = summary[0]['summary_text']
        elif summary_type == 'extractive':
            summary = extractive_summarization(text)
        else:
            return jsonify({"error": "Invalid summary type"}), 400

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)