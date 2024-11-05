from langchain_community.embeddings import OllamaEmbeddings
import json
import numpy as np,os
# llm = Ollama(model="llama3")


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Function to clean the data
def clean_data(data):
    cleaned_data = []
    for study in data:
        cleaned_study = study.copy()
        cleaned_questions = []
        for questions_str in study['questions']:
            try:
                # Try to parse the JSON string
                questions_list = json.loads(questions_str)
                if isinstance(questions_list, list):
                    for question_item in questions_list:
                        # Ensure each item in the list is a dictionary with the required keys
                        if isinstance(question_item, dict) and 'question' in question_item and 'part_of_abstract' in question_item and 'answer' in question_item:
                            cleaned_questions.append(question_item)
            except json.JSONDecodeError:
                # Skip any improperly formatted JSON strings
                continue
        cleaned_study['questions'] = [json.dumps(cleaned_questions)]
        cleaned_data.append(cleaned_study)
    return cleaned_data


def dump_formatted_questions(file_name, data):
    directory='DataFormatting-2'
    # Ensure the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Full path to the JSON file
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as file:
        json.dump(data, file,indent=2)



# Step 1: Read the JSON data from the file
for i in range(3):
    file_path = f"/Users/ykale/Documents/Dev/koios/Koios/LLM_GeneratedQuestions-1/persona_{i}_questions.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Step 2: Clean the data
    cleaned_data = clean_data(data)
    file_name = f"questions_{i}.json"
    print(f"Done with persona:{i} ")
    dump_formatted_questions(file_name, cleaned_data)
