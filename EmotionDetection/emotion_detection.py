"""Emotion detection using the Watson NLP service."""

import json

import requests


def emotion_detector(text_to_analyze):
    """Return emotion scores and the highest-scoring emotion."""
    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
    }
    input_json = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(url, headers=headers, json=input_json)
    response_dict = json.loads(response.text)
    emotions = response_dict["emotionPredictions"][0]["emotion"]
    scores = {
        emotion: emotions[emotion]
        for emotion in ("anger", "disgust", "fear", "joy", "sadness")
    }
    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores
