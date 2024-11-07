from langchain_community.embeddings import OllamaEmbeddings
import json
import numpy as np,os
import csv,glob
ollama_emb = OllamaEmbeddings(model="llama3:latest", base_url="http://localhost:11434")

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
   
def question_embed(file_path):
    data = read_json(file_path)
    print("Data",data['study_id'])
    #print("Data type:", type(data))
    #print("Data type:", type(data))
    # Create a array that stored the questions id and embedding vectore of question
    question_embed_data = []
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                #if isinstance(sub_value, dict):
                if 'question' in sub_value and 'question' in sub_value:
                    #print("This part is embedded==>",sub_value['question'])
                    #question_embed_data.append(sub_value['question'])
                    #question_embed_data.append
                    #print("\n",(sub_value['abstract_part']))
                    #question_embed_data.append(sub_value['answer'])
                    question_embed_data.append((sub_key, sub_value['question'],ollama_emb.embed_query(sub_value['question'])))

    return question_embed_data

        
def convert_to_csv(filename,data):
    directory='QuestionEmbedding-4'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
    print("Data has been written to output csv file")


# Directory containing the formatted questions
input_directory = '/Users/ykale/Documents/Dev/koios/Koios/FormattedQuestions-3'

# Iterate through all JSON files in the directory
for json_file in glob.glob(os.path.join(input_directory, '*.json')):
    data = question_embed(json_file)
    base_name = os.path.basename(json_file)
    csv_filename = f"{os.path.splitext(base_name)[0]}.csv"
    convert_to_csv(csv_filename,data)

    