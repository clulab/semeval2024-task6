import json
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
import transformers
import torch
# https://colab.research.google.com/drive/1SQmK0GYz34RGVlOnL5YMkdm7hXD6OjQT?usp=sharing#scrollTo=m8RwW7Axcu9E
model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-7b-hf

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        list: An array containing the model's response(s).
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    return [sequence['generated_text'] for sequence in sequences]



# Load the data from the JSON file
with open('val.model-agnostic.json', 'r') as file:
    data = json.load(file)

# Initialize the Sentence Transformer model
model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)

# Set a similarity threshold
similarity_threshold = 0.925

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# function to generate definitions - not using context
def generate_definitions(text, context):
    # Should implement method to generate definitions
    # This could be an API call to Llama2 or another service
    prompt = 'Give me 3 defeinitions for ' + text + ' in an array.'
    return get_llama_response(prompt)

# Function to calculate sensibility score (Placeholder)
def calculate_sensibility_score(source, definitions):
    # Placeholder for sensibility score calculation
    # This function should return a sensibility score
    pass

# Initialize list to store DM task data
dm_task_data = []

# Process only DM data types
for datapoint in data:
    if datapoint['task'] == 'DM':
        source = preprocess_text(datapoint['src'])
        hypothesis = preprocess_text(datapoint['hyp'])
        target = preprocess_text(datapoint['tgt'])

        # Generate definitions for the source text
        definitions = generate_definitions(source)
        
        # Calculate the sensibility score for the source text with definitions
        sensibility_score = calculate_sensibility_score(source, definitions)

        # Concatenate source and hypothesis, and calculate new sensibility score
        source_hyp_text = f"{source} {hypothesis}"
        new_sensibility_score = calculate_sensibility_score(source_hyp_text, definitions)

        # Encode the concatenated texts to get embeddings
        source_hyp_embedding = model.encode(source_hyp_text, convert_to_tensor=True)
        source_tgt_embedding = model.encode(f"{source} {target}", convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(source_hyp_embedding, source_tgt_embedding)

        # Determine if the hypothesis is a hallucination
        if similarity < similarity_threshold or new_sensibility_score < sensibility_score:
            predicted_label = "Hallucination"
        else:
            predicted_label = "Not Hallucination"

        true_label = datapoint['label']

        # Append data to DM task list
        dm_task_data.append({
            'source_hyp_embedding': source_hyp_embedding,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'sensibility_score': sensibility_score,
            'new_sensibility_score': new_sensibility_score
        })

# Code for accuracy calculation and results reporting for the DM task
true_labels = [item['true_label'] for item in dm_task_data]
predicted_labels = [item['predicted_label'] for item in dm_task_data]

# Calculate accuracy for the DM task
accuracy = accuracy_score(true_labels, predicted_labels)

# Print results for the DM task
print(f"DM Task Accuracy: {accuracy}")
