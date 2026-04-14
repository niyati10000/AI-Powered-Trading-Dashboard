import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Lazy loading - model and tokenizer will be loaded only when needed
tokenizer = None
model = None
labels = ["negative", "neutral", "positive"]

def _load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading FinBERT sentiment model...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.to(device)
        model.eval()
        print("Model loaded successfully!")

# ✅ FINAL FUNCTION (ONLY ONE)
def get_sentiment(text):
    _load_model()  # Load model if not already loaded
    
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    label = labels[pred]

    return label.upper(), confidence