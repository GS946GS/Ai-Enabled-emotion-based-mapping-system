"""
emotion_model.py
Loads custom HuggingFace RoBERTa emotion classifier.

Functions:
- load_emotion_model()
- predict_emotion(text)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EMOTION_MODEL_DIR = r"C:\Buildathon\final_emotion_model"

ID2LABEL = {
    0:'anger',1:'anxious',2:'bored',3:'calm',4:'confused',5:'depressed',
    6:'excited',7:'fear',8:'frustrated',9:'grateful',10:'happy',11:'lonely',
    12:'motivated',13:'relieved',14:'sad',15:'stress',16:'surprised',17:'tired'
}
LABEL2ID = {v:k for k,v in ID2LABEL.items()}

def load_emotion_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_DIR)

    # Override labels mapping
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    model.to(device)
    model.eval()
    return tokenizer, model, device

emotion_tokenizer, emotion_model, emotion_device = load_emotion_model()

def predict_emotion(text: str):
    if not text or text.strip() == "":
        return {"emotion": None, "confidence": 0.0}

    inp = emotion_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(emotion_device)

    with torch.no_grad():
        out = emotion_model(**inp)
        probs = F.softmax(out.logits, dim=1)[0]
        idx = int(torch.argmax(probs))

    return {
        "emotion": ID2LABEL[idx],
        "confidence": float(probs[idx])
    }
