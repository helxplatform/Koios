from langchain_community.embeddings import OllamaEmbeddings
import json
import os,re

# llm = Ollama(model="llama3")
ollama_emb = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434")


def dump_formatted_questions(file_name, data):
    directory='FormattedQuestions-3'
    # Ensure the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Full path to the JSON file
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as file:
        json.dump(data, file,indent=2)


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def study_question_formatting_test(file_path):
    data = read_json(file_path)
    study_proc =0
    study_not_proc=0
    for study_no in range(len(data)):
        study_data = data[study_no]
        study_number = study_data["study_name"]
        persona = study_data["user_persona"]
        questions = {}
        if "question" in study_data["questions"][0]:
            #print(study_data['questions'][0])
            #print("\n\nThe data type is ",type(study_data['questions'][0]))
            json_str = study_data['questions'][0].split('\n\n')[-1]
            question_data1 = json.loads(json_str)
            for i, question_data in enumerate(question_data1):
                question_id = f"{study_number}_{persona}_{i+1}"
                questions[question_id] = {
                    "question": question_data["question"],
                    "abstract_part": question_data["part_of_abstract"],
                    "answer": question_data["answer"]
                }
            #print("++++++++ Done with ==>", study_number)
            formatted_data = {"study_id": study_number, "Questions": questions}
            file_name = f"{study_number}_{persona}_questions.json"
            dump_formatted_questions(file_name, formatted_data)
            study_proc = study_proc +1
        else:
            #print("--- Skipped ==>", study_number)
            study_not_proc = study_not_proc + 1
    #print(study_proc,study_not_proc)
    print(f"Processed study {study_proc}, Skipped Study:{study_not_proc}")
    return 0


no_of_personas = 3
for i in range(no_of_personas):
    file_path = f"/Users/ykale/Documents/Dev/koios/Koios/DataFormatting-2/questions_{i}.json"
    print("\n")
    study_question_formatting_test(file_path)