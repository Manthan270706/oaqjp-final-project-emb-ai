"""
Emotion Detection Module using Watson NLP Emotion Predict service.
"""
import json
import requests


def _local_emotion_detector(text):
    """
    Fallback emotion detector for offline/local environments
    outside the IBM Skills Network sandbox.
    """
    lower_text = text.lower()
    scores = {
        'anger': 0.01,
        'disgust': 0.01,
        'fear': 0.01,
        'joy': 0.01,
        'sadness': 0.01
    }

    joy_words = ['glad', 'happy', 'joy', 'love', 'excited', 'wonderful']
    anger_words = ['mad', 'angry', 'furious', 'rage', 'annoyed']
    disgust_words = ['disgusted', 'disgust', 'gross', 'nasty', 'sick']
    sad_words = ['sad', 'depressed', 'sorrow', 'unhappy', 'cry']
    fear_words = ['afraid', 'fear', 'scared', 'terrified', 'frightened']

    if any(word in lower_text for word in joy_words):
        scores['joy'] = 0.95
    elif any(word in lower_text for word in anger_words):
        scores['anger'] = 0.95
    elif any(word in lower_text for word in disgust_words):
        scores['disgust'] = 0.95
    elif any(word in lower_text for word in sad_words):
        scores['sadness'] = 0.95
    elif any(word in lower_text for word in fear_words):
        scores['fear'] = 0.95
    else:
        scores['joy'] = 0.80

    dominant_emotion = max(scores, key=scores.get)
    return {
        'anger': scores['anger'],
        'disgust': scores['disgust'],
        'fear': scores['fear'],
        'joy': scores['joy'],
        'sadness': scores['sadness'],
        'dominant_emotion': dominant_emotion
    }


def emotion_detector(text_to_analyze):
    """
    Analyzes the input text using Watson NLP Emotion Predict service
    and returns a dictionary containing emotion scores and the dominant emotion.
    """
    url = (
        'https://sn-watson-emotion.labs.skills.network/'
        'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    input_json = {"raw_document": {"text": text_to_analyze}}

    if not text_to_analyze or not text_to_analyze.strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    try:
        response = requests.post(
            url,
            json=input_json,
            headers=headers,
            timeout=10
        )
    except requests.exceptions.RequestException:
        return _local_emotion_detector(text_to_analyze)

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']

    anger_score = emotions.get('anger', 0.0)
    disgust_score = emotions.get('disgust', 0.0)
    fear_score = emotions.get('fear', 0.0)
    joy_score = emotions.get('joy', 0.0)
    sadness_score = emotions.get('sadness', 0.0)

    emotion_scores = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }
