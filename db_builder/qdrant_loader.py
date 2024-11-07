
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import pandas as pd
import glob,os,time

client = QdrantClient(url="http://localhost:54713/",timeout=120)
'''client.create_collection(
    collection_name="LLM_generated_question_collection",
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)'''


#Read the question no and it's corresponding vector embedding.
def read_vector_embed(collect_name):
    input_directory ="/Users/ykale/Documents/Dev/koios/Koios/QuestionEmbedding-4"
    records = []
    main_index= 1
    for count,filename in enumerate(glob.glob(os.path.join(input_directory, '*.csv'))):
        if main_index % 5 == 0:
            print("Adding delay of 3 seconds...")
            time.sleep(3)
        # Load the CSV file
        print("Working on",filename)
        df = pd.read_csv(filename, header=None)
        df.columns = ['question_id','question', 'embedding']
        # Convert embeddings from string to list of floats
        df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
        for index, row in df.iterrows():
            record = {
                'id': main_index,  # Incremental integer id
                'vector': row['embedding'],
                'payload': {'question_id': row['question_id'],'question':row['question']}  # Store question_id in payload
                  }
            main_index = main_index + 1
            records.append(record)
            # Upload records to Qdrant
        client.upsert(
            collection_name=collect_name,
            wait=True,
            points=records
        )
        print(f"Inserted all the records Successfully")
    return records


def create_a_collection_qdrant(collect_name):
    collection_name = collect_name
    vector_size = 4096 # Assuming all vectors are of the same length
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance='Cosine')  #'Cosine' to 'Euclidean' or 'Dot' as needed
    )


collect_name = 'test_collection2'
create_a_collection_qdrant(collect_name)
records = read_vector_embed(collect_name)
#insert_data_qdrant(records)

