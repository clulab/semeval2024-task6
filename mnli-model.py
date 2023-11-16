import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score
from trial_test import preprocess_text

# Load the data from the JSON file
with open('val.model-agnostic.json', 'r') as file:
    data = json.load(file)

# Initialize the BART model and tokenizer for MNLI entailment
model_name = 'facebook/bart-large-mnli'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Encode the concatenated texts to get logits for entailment
    inputs = tokenizer(source_hyp_text, source_tgt_text, return_tensors="pt", padding=True, truncation=True)
    logits = model(**inputs).logits

    # Get the model's prediction for entailment
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to binary classification labels
    if predicted_class == 0:
        predicted_label = "Not Hallucination"
    else:
        predicted_label = "Hallucination"

    true_label = datapoint['label']

    # Append data to the appropriate task
    tasks[task].append({
        'logits': logits,
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
    print("\n")

print(f"Total Accuracy: {total_accuracy}")
