import numpy as np

fear_words = ["crash", "fear", "panic", "recession", "uncertainty"]

def compute_fear_index(text):
    count = sum([1 for word in fear_words if word in text])
    return count / len(fear_words)

def create_feature_vector(sentiment, fear):
    return np.array([sentiment, fear])