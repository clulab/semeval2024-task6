import json
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the JSON file
with open('val.model-agnostic.json', 'r') as file:
    data = json.load(file)

# Initialize the Sentence Transformer model
model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)

# Set a similarity threshold (you can adjust this value)
similarity_threshold = 0.925

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Initialize dictionaries to store data for each task
tasks = {'MT': [], 'DM': [], 'PG': []}

# Loop through each data point and group by task
for datapoint in data:
    task = datapoint['task']
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

    # Append data to the appropriate task
    tasks[task].append({
        'source_hyp_embedding': source_hyp_embedding,
        'source_tgt_embedding': source_tgt_embedding,
        'true_label': true_label,
        'predicted_label': predicted_label
    })

# Initialize dictionaries to store results for each task
task_results = {}

# Calculate accuracy for each task and total accuracy
total_true_labels = []
total_predicted_labels = []

for task, task_data in tasks.items():
    true_labels = [item['true_label'] for item in task_data]
    predicted_labels = [item['predicted_label'] for item in task_data]

    # Calculate accuracy for the current task
    accuracy = accuracy_score(true_labels, predicted_labels)

    task_results[task] = {
        'accuracy': accuracy,
        # 'classification_report': classification_report(true_labels, predicted_labels, target_names=['Hallucination', 'Not Hallucination'])
    }

    # Accumulate true and predicted labels for total accuracy
    total_true_labels.extend(true_labels)
    total_predicted_labels.extend(predicted_labels)

# Calculate total accuracy
total_accuracy = accuracy_score(total_true_labels, total_predicted_labels)

# Print results for each task and total accuracy
for task, results in task_results.items():
    print(f"Task: {task}")
    print(f"Accuracy: {results['accuracy']}")
    # print("Classification Report:")
    # print(results['classification_report'])
    print("\n")

print(f"Total Accuracy: {total_accuracy}")
