import json
import random

def extract_random_dm_examples(file_path, num_examples=5):
    try:
        # Read the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Filter data for 'DM' tasks
        dm_data = [entry for entry in data if entry.get('task') == 'DM']

        # Randomly select num_examples from the filtered data
        random_dm_examples = random.sample(dm_data, min(num_examples, len(dm_data)))

        return random_dm_examples

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON data.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage on trial data
file_path = 'trial-v1.json'
random_examples = extract_random_dm_examples(file_path)

# Display the random examples
for example in random_examples:
    print(example)
