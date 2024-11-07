import json,os
def read_json(file_path):
    with open(file_path, 'r') as stream:
        data = json.load(stream)
    return data


def dump_formatted_questions(file_name, data):
    directory='formatted_questions'
    # Ensure the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Full path to the JSON file
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as file:
        json.dump(data, file,indent=2)


def extract_info(study):
    count=0
    count1=0
    study_processed=[]
    study_not_processed=[]
    for index in range(len(study)):
        print("+++++++\n",study[index]['study_name'])
        for questions in study[index]['questions']:
            print("questions-------->",type(questions))
            cleaned_questions=questions.replace("[", "").replace("]", "").replace("}{", "},{")
            cleanned_questions = "["+cleaned_questions+"]"
            print("\ncleanned_questions========",cleanned_questions)
            #data1= eval(data2)
            try:
                json_formatted = json.loads(cleanned_questions)
                for item in json_formatted:
                    print(f"===>Question: {item['question']}")
                    print(f"===>Part of Abstract: {item['part_of_abstract']}")
                    print(f"==>Answer: {item['answer']}")
                    print() 
                count1= count1+1
            except:
                print("Exited study no",study[index]['study_name'])
                count = count + 1
    return(count1,count)


def json_study_formatting(study):
    count=0
    count1=0
    study_processed=[]
    study_not_processed=[]
    for study_no in range(len(study)):
        study_data = study[study_no]
        study_number = study_data["study_name"]
        persona = study_data["user_persona"]
        questions = {}
        for questions in study[study_no]['questions']:
            cleaned_questions=questions.replace("[", "").replace("]", "").replace("}{", "},{")
            cleanned_questions = "["+cleaned_questions+"]"
            #try:
            json_formatted = json.loads(cleanned_questions)
            #print(json_formatted)
            i=0
            for question_data in json_formatted:
                question_id = f"{study_number}_{persona}_{i+1}"
                i=i+1
                print("data['question']",question_data['question'])
                print("data['question']",question_data['part_of_abstract'])
                print("data['question']",question_data['answer'])
                #questions[question_id] = {"question": question_data['question'],
                #                            "abstract_part": question_data['part_of_abstract'],
                #                            "answer": question_data['answer']}
            print("++++++++++++++ Done with", study_number)
            formatted_data = {"study_id": study_number, "Questions": questions}
            file_name = f"{study_number}_{persona}_questions.json"
            dump_formatted_questions(file_name, formatted_data)
            count1= count1+1
        #except:
        print("Exited study no",study[study_no]['study_name'])
        count = count + 1
    return(count1,count)
for i in range(1):
    file_path = f"/Users/ykale/Documents/Dev/koios/Koios/LLM_generated_questions/persona_{i}_questions.json"
    data = read_json(file_path)
    print(json_study_formatting(data))

