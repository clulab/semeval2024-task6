import json
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the JSON file
with open('trial-v1.json', 'r') as file:
    data = json.load(file)

# Initialize the Sentence Transformer model
model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)

# Initialize lists to store embeddings and labels
source_hyp_embeddings = []
source_tgt_embeddings = []
true_labels = []
predicted_labels = []

# Set a similarity threshold
similarity_threshold = 0.9

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Loop through each data point
for datapoint in data:
    source = preprocess_text(datapoint['src'])
    hypothesis = preprocess_text(datapoint['hyp'])
    target = preprocess_text(datapoint['tgt'])

    # Concatenate source and hypothesis
    source_hyp_text = f"{source} {hypothesis}"
     # Concatenate source and target
    source_tgt_text = f"{source} {target}"

    # Encode the concatenated texts to get embeddings
    source_hyp_embedding = model.encode(source_hyp_text, convert_to_tensor=True)
    source_tgt_embedding = model.encode(source_tgt_text, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(source_hyp_embedding, source_tgt_embedding)

    # Assign binary classification based on similarity threshold
    if similarity > similarity_threshold:
        predicted_label = "Not Hallucination"
    else:
        predicted_label = "Hallucination"

    true_label = datapoint['label']

    # Append embeddings and labels
    source_hyp_embeddings.append(source_hyp_embedding)
    source_tgt_embeddings.append(source_tgt_embedding)
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Generate classification report
classification_rep = classification_report(true_labels, predicted_labels, target_names=['Hallucination', 'Not Hallucination'])

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)