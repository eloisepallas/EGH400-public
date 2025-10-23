import nltk
import string
from nltk.tokenize import word_tokenize
from nrclex import NRCLex
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt')

# Raw input (used for RoBERTa)
text_input = "Course Evals and Rate my Professor maybe venting, maybe advice seeking. I’m a first time faculty member. I taught in grad school but just got my first set of course evals at my new school. Plus some rate my professors. Y’all, someone gave me a review and admitted “I never attended class”. This was a RMP but it’s just wild that students who never even came to class can affect my job. My official evaluation scores were also brought down by students saying class was boring because it was all lecture or “prof just read from the slides”. BUT IT WAS A LECTURE COURSE! If I did more engagement, they’d all be complaining I didn’t teach.  But for advice, how seriously should I be taking these comments for the future? I want to improve as a teacher and plan on incorporating more active learning and interaction. But I’m worried these negative comments are from students who never came to class or were zoned out"

# Clean text for NRCLex (lowercase, remove punctuation, tokenize)
cleaned_text = text_input.lower().translate(str.maketrans('', '', string.punctuation))
tokens = word_tokenize(cleaned_text)
nrc_ready_text = " ".join(tokens)

# NRCLex output
nrc_text = NRCLex(nrc_ready_text)
nrc_emotions = nrc_text.raw_emotion_scores
nrc_total = sum(nrc_emotions.values()) or 1
nrc_normalized = {emotion: score / nrc_total for emotion, score in nrc_emotions.items()}

# RoBERTa (GoEmotions) output
classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5)
roberta_results = classifier(text_input)[0]
roberta_emotions = {entry['label'].lower(): entry['score'] for entry in roberta_results}

# Combine and align emotion keys
all_emotions = sorted(set(nrc_normalized.keys()).union(roberta_emotions.keys()))
scores = {
    'NRCLex': [nrc_normalized.get(e, 0) for e in all_emotions],
    'RoBERTa (GoEmotions)': [roberta_emotions.get(e, 0) for e in all_emotions]
}

# Plot
x = np.arange(len(all_emotions))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, scores['NRCLex'], width, label='NRCLex')
ax.bar(x + width/2, scores['RoBERTa (GoEmotions)'], width, label='RoBERTa (GoEmotions)')

ax.set_ylabel('Emotion Score', fontsize=18)
ax.set_title(f"Emotion Detection on sample post: '{text_input}'", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(all_emotions, ha='right', rotation=45, fontsize=12)
ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
