# Sentiment Analysis API

## Description
This is a sentiment analysis API developed with FastAPI. The API receives a list of products and their comments, processes the data, and returns information about emotions, polarity, keywords, and product ranking scores.

## Features
- **Emotion classification** in comments using the "j-hartmann/emotion-english-distilroberta-base" model. This model identifies various emotions such as **joy, anger, sadness, surprise, and more**, providing deeper insights into user sentiment beyond just positive or negative classification.
- **Sentiment polarity analysis** (positive/negative) using the "distilbert-base-uncased-finetuned-sst-2-english" model.
- **Keyword extraction** using the RAKE algorithm to identify the most relevant phrases in the comments.
- **Calculation of a ranking score** based on the proportion of positive and negative comments.

## Technologies Used
- Python
- FastAPI
- scikit-learn
- pandas
- transformers
- torch
- joblib
- rake-nltk
- nltk
- uvicorn

## Installation and Setup

### Requirements
Ensure you have Python installed (version 3.8+).

### Installing Dependencies
```sh
pip install fastapi uvicorn scikit-learn pandas transformers torch joblib rake-nltk nltk
```

### Downloading Additional NLTK Resources
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## How to Run the API

```sh
uvicorn main:app --host 0.0.0.0 --port 3000
```

The API will be available at: `http://0.0.0.0:3000`

## Available Endpoints

### `GET /`
Test route to check if the API is working correctly.

#### Successful Response:
```json
{
    "message": "Sentiment Analysis API. Use the /analyze endpoint to submit data."
}
```

### `POST /analyze`
Receives a list of products and their comments, performs sentiment analysis, and returns the results.

#### Example Request:
```json
{
    "products": [
        {
            "product_name": "Smartphone X",
            "comments": [
                {"comment": "The camera is amazing!", "date": "2025-03-22"},
                {"comment": "Battery life is too short...", "date": "2025-03-21"}
            ]
        }
    ]
}
```

#### Example Response:
```json
{
    "results": [
        {
            "product_name": "Smartphone X",
            "emotion_distribution": {
                "joy": 1,
                "anger": 1
            },
            "top_keywords": ["camera amazing", "battery life too short"],
            "pos_count": 1,
            "neg_count": 1,
            "ranking_score": 0.0,
            "comments": [
                {
                    "original": "The camera is amazing!",
                    "cleaned": "camera amazing",
                    "emotion": "joy",
                    "polarity": "POSITIVE",
                    "date": "2025-03-22"
                },
                {
                    "original": "Battery life is too short...",
                    "cleaned": "battery life too short",
                    "emotion": "anger",
                    "polarity": "NEGATIVE",
                    "date": "2025-03-21"
                }
            ]
        }
    ]
}
```

## Possible Future Improvements
- Support for languages other than English.
- Implementation of caching to optimize API execution.
- More in-depth analysis with models specifically trained for product reviews.

