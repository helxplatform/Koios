"""
Qdrant Question Generation and Upload Pipeline

This script provides a complete pipeline for:
1. Generating natural language questions from study descriptions using LLM
2. Creating payloads with question embeddings
3. Uploading embeddings to Qdrant vector database

Author: YSK
Date: 2025
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
import json
import csv
import requests
import pandas as pd
from typing import List, Dict
import re
import time
from pathlib import Path

# ============================================================================
# CONFIGURATION - Load from .env file
# ============================================================================

def load_env_config():
    """Load configuration from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    config = {}

    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert to appropriate type
                    if value.isdigit():
                        value = int(value)
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value == '':
                        value = None

                    config[key] = value

    return config

# Load configuration from .env
_config = load_env_config()

# File paths
INPUT_STUDIES_FILE = _config.get('INPUT_STUDIES_FILE', "99_select_studies.json")
INPUT_PERSONAS_FILE = _config.get('INPUT_PERSONAS_FILE', "personas.csv")
OUTPUT_QUESTIONS_FILE = _config.get('OUTPUT_QUESTIONS_FILE', "generated_questions_payloads.json")
INPUT_EMBEDDINGS_FILE = _config.get('INPUT_EMBEDDINGS_FILE', "generated_questions_payloads_with_embeddings.json")

# LLM configuration
LLM_BASE_URL = _config.get('LLM_BASE_URL', "http://localhost:52236")
LLM_MODEL_NAME = _config.get('LLM_MODEL_NAME', "google/gemma-3-12b-it")

# Embedding configuration
EMBEDDING_BASE_URL = _config.get('EMBEDDING_BASE_URL', "http://localhost:11434")
EMBEDDING_MODEL_NAME = _config.get('EMBEDDING_MODEL_NAME', "bge-m3")
EMBEDDING_BATCH_SIZE = _config.get('EMBEDDING_BATCH_SIZE', 100)
EMBEDDING_SAVE_INTERVAL = _config.get('EMBEDDING_SAVE_INTERVAL', 50)

# Qdrant configuration
QDRANT_URL = _config.get('QDRANT_URL', "http://localhost:60850")
QDRANT_COLLECTION_NAME = _config.get('QDRANT_COLLECTION_NAME', "Collection_name")
VECTOR_SIZE = _config.get('VECTOR_SIZE', 1024)
DISTANCE_METRIC = _config.get('DISTANCE_METRIC', "Cosine")  # Options: Cosine, Euclid, Dot
UPLOAD_BATCH_SIZE = _config.get('UPLOAD_BATCH_SIZE', 100)

# Generation parameters
START_PERSONA = _config.get('START_PERSONA', 16)
MAX_STUDIES = _config.get('MAX_STUDIES', None)  # Set to a number to limit, None for all
MAX_PERSONAS = _config.get('MAX_PERSONAS', None)  # Set to a number to limit, None for all
SAVE_INTERVAL = _config.get('SAVE_INTERVAL', 100)  # Save progress every N questions

# ============================================================================

def connect_and_display_records(qdrant_url=QDRANT_URL, collection_name=QDRANT_COLLECTION_NAME):
    """Display sample records from a Qdrant collection."""
    client = QdrantClient(url=qdrant_url)

    try:
        # Get collection info to understand vector dimensions
        collection_info = client.get_collection(collection_name)
        print(f"Collection info: {collection_info}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
        print("-" * 50)

        records = client.scroll(
            collection_name=collection_name,
            limit=2,
            with_payload=True,
            with_vectors=True
        )

        print(f"First 2 records from collection '{collection_name}':")
        print("-" * 50)

        for i, record in enumerate(records[0], 1):
            print(f"Record {i}:")
            print(f"ID: {record.id}")
            print(f"Payload: {record.payload}")
            if record.vector:
                print(f"Vector dimensions: {len(record.vector)}")
                print(f"First 5 vector values: {record.vector[:5]}")
            print("-" * 30)

        return collection_info.config.params.vectors.size, collection_info.config.params.vectors.distance

    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None, None

def create_bdc_collection(qdrant_url=QDRANT_URL, new_collection_name=QDRANT_COLLECTION_NAME):
    """Create a new Qdrant collection with the same configuration as the reference collection."""
    client = QdrantClient(url=qdrant_url)

    try:
        # First get the vector dimensions from the existing collection
        vector_size, distance_metric = connect_and_display_records()

        if vector_size is None:
            print("Could not get vector dimensions from existing collection")
            return

        print(f"\nCreating new collection '{new_collection_name}' with:")
        print(f"Vector size: {vector_size}")
        print(f"Distance metric: {distance_metric}")

        # Create the new collection with same configuration
        client.create_collection(
            collection_name=new_collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_metric
            )
        )

        print(f"Successfully created collection '{new_collection_name}'")

        # Verify the collection was created
        collections = client.get_collections()
        print(f"Available collections: {[c.name for c in collections.collections]}")

    except Exception as e:
        print(f"Error creating collection: {e}")

class QuestionGenerationPipeline:
    """Pipeline for generating natural language questions from study descriptions using LLM."""

    def __init__(self, llm_base_url=LLM_BASE_URL, model_name=LLM_MODEL_NAME):
        self.llm_base_url = llm_base_url
        self.model_name = model_name

    def load_studies(self, json_file_path: str) -> List[Dict]:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def load_personas(self, csv_file_path: str) -> List[str]:
        personas = []
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    personas.append(row[1])
        return personas

    def generate_questions_with_llm(self, description: str, persona: str) -> List[str]:
        system_prompt = (
            f"You are generating natural search queries from the perspective of: {persona}. "
            "These queries will be embedded using semantic search to retrieve this specific study abstract when real users ask similar questions. "
            "Your goal is to anticipate how THIS persona would naturally phrase questions that this study could answer. "
            "\n"
            "Generate exactly 5 diverse, natural-language search queries that would lead users to THIS specific study. "
            "\n"
            "Critical Requirements: "
            "1. NATURAL LANGUAGE: Write as real users would ask - use complete thoughts, not keyword lists. Mix questions and statements naturally. "
            "2. DIVERSE USER INTENTS: Cover different reasons someone would seek this study: "
            "   - Intent 1: Finding datasets by disease/condition (e.g., 'where can I find data on...')"
            "   - Intent 2: Methodology/design questions (e.g., 'what studies use longitudinal cohort design for...')"
            "   - Intent 3: Specific features (e.g., 'datasets with genetic data and family history for...')"
            "   - Intent 4: Population/eligibility (e.g., 'studies that include multi-generational participants...')"
            "   - Intent 5: Access/practical concerns (e.g., 'available cardiovascular datasets with phenotype and genotype data')"
            "3. PERSONA-AUTHENTIC: Frame each query using terminology and concerns this specific persona would have "
            "4. STUDY-SPECIFIC: Include 2-3 unique identifiers per query (study name, disease, sample size, specific methodology, unique features) that distinguish this study from others "
            "5. SEMANTIC VARIETY: Use synonyms and different phrasings - avoid repeating the same keywords across all 5 queries "
            "6. CONVERSATIONAL: Write how people talk to chatbots, not how they write academic searches "
            "7. NO QUESTION MARKS: Write queries as statements or questions without ending punctuation "
            "8. NO MARKDOWN: Do not use markdown code blocks like ```json or ``` in your response "
            "9. NO EXTRA TEXT: Do not include any explanations, notes, or text outside the JSON array "
            "\n"
            "Format Requirements: "
            "- Return ONLY a valid JSON array of exactly 5 strings "
            "- The array must start with [ and end with ] "
            "- Each string must be enclosed in double quotes "
            "- Strings must be separated by commas "
            "- Do NOT wrap the array in markdown code blocks "
            "- Do NOT add any text before or after the array "
            "\n"
            "Example of GOOD queries (natural, diverse intents, conversational):\n"
            "[\"Where can I find longitudinal heart disease data that follows families across multiple generations\", "
            "\"I need cardiovascular risk factor datasets with both genetic and clinical data from large cohorts\", "
            "\"What studies have echocardiography imaging paired with long-term health outcomes for heart disease\", "
            "\"Looking for datasets that track blood pressure cholesterol and smoking in the same participants over decades\", "
            "\"Are there any large family-based studies with genome-wide genotyping for cardiovascular disease research\"]\n"
            "\n"
            "BAD example (keyword lists, no diversity):\n"
            "[\"Framingham Heart Study cardiovascular risk factors 5209 participants\", "
            "\"Framingham study blood pressure cholesterol\", "
            "\"Framingham three generation family study\", "
            "\"Framingham genome analysis\", "
            "\"Framingham echocardiography imaging\"]"
        )

        human_prompt = f"The study abstract starts here after : \n {description}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        try:
            response = requests.post(f"{self.llm_base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()

            generated_text = result['choices'][0]['message']['content']

            # Parse the generated questions (assuming they're in a list format)
            questions = self.parse_questions_from_response(generated_text)

            return questions[:5]  # Return only 5 questions as requested

        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    def parse_questions_from_response(self, response_text: str) -> List[str]:
        # Try to extract questions from the response
        # Remove any extra formatting and split by common separators
        cleaned_text = response_text.strip()

        # If it looks like a Python list, try to evaluate it safely
        if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
            try:
                # Use regex to extract quoted strings from the list
                questions = re.findall(r'"([^"]*)"', cleaned_text)
                if not questions:
                    questions = re.findall(r"'([^']*)'", cleaned_text)
                return questions
            except:
                pass

        # Fallback: split by common separators
        if ',' in cleaned_text:
            questions = [q.strip().strip('"\'') for q in cleaned_text.split(',')]
        elif '\n' in cleaned_text:
            questions = [q.strip().strip('"\'') for q in cleaned_text.split('\n') if q.strip()]
        else:
            questions = [cleaned_text.strip().strip('"\'')]

        # Clean up questions
        questions = [q for q in questions if q and len(q) > 10]
        return questions

    def create_qdrant_payload(self, study_data: Dict, persona: str, questions: List[str], persona_num: int, start_question_num: int) -> List[Dict]:
        """Create payload structure matching existing Qdrant collection format with correct question_id numbering"""
        payloads = []
        for i, question in enumerate(questions):
            # Create question_id in format: study_id_personanumber_questionnumber (e.g., phs003529.v1.p1_1_1)
            question_number = start_question_num + i
            # Use StudyId as study_id for the new JSON structure
            study_id = study_data.get('StudyId', '')
            question_id = f"{study_id}_{persona_num}_{question_number}"

            # Payload structure matching existing 'questions_on_study_abstract_bge' collection
            payload = {
                'question': question,  # Main question text (matches existing format)
                'question_id': question_id,  # Unique ID (format: study_id_personanumber_questionnumber)
                # Additional metadata (not in original but useful for filtering/search)
                'study_id': study_id,
                'study_name': study_data.get('StudyName', ''),
                'persona': persona,
                'description': study_data.get('Description', ''),
                'permalink': study_data.get('Permalink', ''),
                # Placeholder for vector embedding (to be added later)
                'embedding': None
            }
            payloads.append(payload)
        return payloads

    def process_all_studies_and_personas(self, json_file_path: str, csv_file_path: str,
                                        max_studies: int = None, max_personas: int = None,
                                        start_persona: int = 1,
                                        incremental_save: bool = True,
                                        save_interval: int = 100,
                                        output_file: str = "generated_questions_payloads.json") -> List[Dict]:
        studies = self.load_studies(json_file_path)
        personas = self.load_personas(csv_file_path)

        # Limit for testing if specified
        if max_studies:
            studies = studies[:max_studies]
            print(f"Limited to first {max_studies} studies for testing")

        if max_personas:
            personas = personas[:max_personas]
            print(f"Limited to first {max_personas} personas for testing")

        # Skip personas before start_persona
        if start_persona > 1:
            personas = personas[start_persona - 1:]
            print(f"Starting from persona {start_persona}")

        all_payloads = []
        total_combinations = len(studies) * len(personas)
        processed = 0
        last_save_count = 0

        print(f"Processing {len(studies)} studies with {len(personas)} personas...")
        print(f"Total combinations to process: {total_combinations}")
        print(f"Incremental save: {'enabled' if incremental_save else 'disabled'}")
        if incremental_save:
            print(f"Saving every {save_interval} questions")

        for persona_idx, persona in enumerate(personas):
            # Adjust persona_num to account for start_persona offset
            persona_num = persona_idx + start_persona

            print(f"\n{'='*70}")
            print(f"PROCESSING PERSONA {persona_num}/{start_persona + len(personas) - 1}")
            print(f"Persona: {persona[:80]}...")
            print(f"{'='*70}")

            for study_idx, study in enumerate(studies):
                study_id = study.get('StudyId', 'unknown')
                description = study.get('Description', '')

                if not description:
                    print(f"Skipping study {study_id} - no description")
                    continue

                processed += 1
                print(f"Study {study_idx + 1}/{len(studies)}: {study_id} (Overall: {processed}/{total_combinations})")

                questions = self.generate_questions_with_llm(description, persona)

                if questions:
                    # Each persona gets questions numbered 1 to 5 (or however many questions are generated)
                    payloads = self.create_qdrant_payload(study, persona, questions, persona_num, 1)
                    all_payloads.extend(payloads)
                    print(f"  ✓ Generated {len(questions)} questions (Total: {len(all_payloads)})")

                    # Save every save_interval questions
                    if incremental_save and len(all_payloads) - last_save_count >= save_interval:
                        print(f"\n{'='*70}")
                        print(f"SAVING PROGRESS: {len(all_payloads)} questions generated")
                        print(f"New questions since last save: {len(all_payloads) - last_save_count}")
                        self.save_payloads_to_json(all_payloads, output_file)
                        print(f"Saved to {output_file}")
                        print(f"Pausing for 10 seconds...")
                        time.sleep(10)
                        print(f"{'='*70}\n")
                        last_save_count = len(all_payloads)
                else:
                    print(f"  ✗ No questions generated")

        # Final save if there are any remaining unsaved payloads
        if incremental_save and len(all_payloads) > last_save_count:
            print(f"\n{'='*70}")
            print(f"FINAL SAVE: {len(all_payloads)} total questions")
            print(f"New questions since last save: {len(all_payloads) - last_save_count}")
            self.save_payloads_to_json(all_payloads, output_file)
            print(f"Saved to {output_file}")
            print(f"{'='*70}\n")

        print(f"\nTotal payloads created: {len(all_payloads)}")
        return all_payloads

    def save_payloads_to_json(self, payloads: List[Dict], output_file: str = "generated_questions_payloads.json"):
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(payloads, file, indent=2, ensure_ascii=False)
        print(f"Payloads saved to {output_file}")

    def save_payloads_to_csv(self, payloads: List[Dict], output_file: str = "generated_questions_payloads.csv"):
        """Save payloads to CSV for easy viewing"""
        if not payloads:
            print("No payloads to save")
            return

        df = pd.DataFrame(payloads)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Payloads saved to {output_file}")
        print(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

    def run_complete_pipeline(self,
                            json_file_path: str = INPUT_STUDIES_FILE,
                            csv_file_path: str = INPUT_PERSONAS_FILE,
                            output_file: str = OUTPUT_QUESTIONS_FILE,
                            max_studies: int = MAX_STUDIES,
                            max_personas: int = MAX_PERSONAS,
                            start_persona: int = START_PERSONA,
                            save_interval: int = SAVE_INTERVAL,
                            save_csv: bool = True):
        print("Starting Question Generation Pipeline...")
        print("=" * 60)

        # Process studies and personas
        payloads = self.process_all_studies_and_personas(
            json_file_path, csv_file_path, max_studies, max_personas,
            start_persona, incremental_save=True, save_interval=save_interval,
            output_file=output_file
        )

        # Save final results
        self.save_payloads_to_json(payloads, output_file)

        if save_csv:
            self.save_payloads_to_csv(payloads)

        print("\nPipeline completed!")
        print(f"Generated {len(payloads)} question payloads with Qdrant-compatible format")
        print("Next step: Add embeddings to the 'embedding' field before inserting to Qdrant")
        print("Payload structure matches existing collection: 'question' and 'question_id' fields")

        return payloads

def run_question_generation_pipeline():
    """Run the complete question generation pipeline with configured parameters."""
    pipeline = QuestionGenerationPipeline(
        llm_base_url=LLM_BASE_URL,
        model_name=LLM_MODEL_NAME
    )
    return pipeline.run_complete_pipeline()

def test_question_generation_pipeline():
    """Run pipeline with limited data for testing."""
    pipeline = QuestionGenerationPipeline(
        llm_base_url=LLM_BASE_URL,
        model_name=LLM_MODEL_NAME
    )
    return pipeline.run_complete_pipeline(
        max_studies=5,
        max_personas=1,
        save_csv=True
    )

def generate_embeddings(
    input_file: str = OUTPUT_QUESTIONS_FILE,
    output_file: str = INPUT_EMBEDDINGS_FILE,
    embedding_base_url: str = EMBEDDING_BASE_URL,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    save_interval: int = EMBEDDING_SAVE_INTERVAL,
    start_index: int = 0
):
    """
    Generate embeddings for questions using Ollama embeddings.

    Args:
        input_file: Path to input JSON file with questions
        output_file: Path to output JSON file with embeddings
        embedding_base_url: Base URL for embedding service
        embedding_model: Name of the embedding model
        batch_size: Number of questions to process at once
        save_interval: Save after this many embeddings
        start_index: Index to start from (useful for resuming)

    Returns:
        List of questions with embeddings
    """
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        print("Error: langchain_ollama not installed. Install with: pip install langchain-ollama")
        return None

    print(f"Initializing OllamaEmbeddings with model: {embedding_model}")
    print(f"Base URL: {embedding_base_url}")
    ollama_emb = OllamaEmbeddings(model=embedding_model, base_url=embedding_base_url)

    # Load questions
    with open(input_file, 'r', encoding='utf-8') as file:
        questions = json.load(file)

    total_questions = len(questions)
    print(f"\n{'='*70}")
    print(f"Starting embedding generation")
    print(f"Total questions: {total_questions}")
    print(f"Starting from index: {start_index}")
    print(f"Batch size: {batch_size}")
    print(f"Save interval: {save_interval}")
    print(f"{'='*70}\n")

    last_save_index = start_index

    # Process questions in batches
    for i in range(start_index, total_questions, batch_size):
        batch_end = min(i + batch_size, total_questions)
        batch = questions[i:batch_end]

        # Extract question texts for this batch
        question_texts = [q['question'] for q in batch]

        try:
            print(f"Processing questions {i+1}-{batch_end}/{total_questions}...")
            embeddings = ollama_emb.embed_documents(question_texts)

            # Add embeddings to the questions
            for j, embedding in enumerate(embeddings):
                questions[i + j]['embedding'] = embedding

            print(f"  ✓ Generated {len(embeddings)} embeddings")

            # Save incrementally
            if (i + batch_size) - last_save_index >= save_interval or batch_end == total_questions:
                print(f"\n{'='*70}")
                print(f"SAVING PROGRESS: {batch_end}/{total_questions} questions processed")
                print(f"New embeddings since last save: {batch_end - last_save_index}")
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(questions, file, indent=2, ensure_ascii=False)
                print(f"Saved to {output_file}")

                if batch_end < total_questions:
                    print(f"Pausing for 10 seconds...")
                    time.sleep(10)

                print(f"{'='*70}\n")
                last_save_index = batch_end

        except Exception as e:
            print(f"  ✗ Error processing batch {i+1}-{batch_end}: {e}")
            print(f"Saving progress before error...")
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(questions, file, indent=2, ensure_ascii=False)
            raise

    print(f"\n{'='*70}")
    print(f"COMPLETED: All {total_questions} embeddings generated")
    print(f"Output saved to: {output_file}")
    print(f"{'='*70}\n")

    return questions

def upload_embeddings_to_qdrant(
    json_file_path: str = INPUT_EMBEDDINGS_FILE,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION_NAME,
    batch_size: int = UPLOAD_BATCH_SIZE,
    create_if_missing: bool = True,
    vector_size: int = VECTOR_SIZE,
    distance_metric: str = DISTANCE_METRIC
):
    """
    Upload embeddings from JSON file to Qdrant collection.

    Args:
        json_file_path: Path to the JSON file containing question payloads with embeddings
        qdrant_url: URL of the Qdrant instance
        collection_name: Name of the collection to upload to
        batch_size: Number of points to upload in each batch
        create_if_missing: Create collection if it doesn't exist
        vector_size: Size of the embedding vectors
        distance_metric: Distance metric to use (Cosine, Euclid, or Dot)

    Returns:
        Number of successfully uploaded points
    """
    print(f"Loading embeddings from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as file:
        payloads = json.load(file)

    print(f"Loaded {len(payloads)} payloads")

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url)

    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
    except Exception as e:
        if create_if_missing:
            print(f"Collection does not exist. Creating '{collection_name}'...")

            # Map distance metric string to enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT
            }

            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance_map.get(distance_metric, Distance.COSINE)
                    )
                )
                print(f"Successfully created collection '{collection_name}'")
                print(f"Vector size: {vector_size}")
                print(f"Distance metric: {distance_metric}")
            except Exception as create_error:
                print(f"Error creating collection: {create_error}")
                return 0
        else:
            print(f"Collection does not exist: {e}")
            return 0

    # Prepare points for upload
    points = []
    skipped = 0

    for idx, payload in enumerate(payloads):
        # Check if embedding exists
        if 'embedding' not in payload or payload['embedding'] is None:
            skipped += 1
            continue

        # Extract embedding and create payload without it (Qdrant stores vectors separately)
        embedding = payload['embedding']

        # Create payload for Qdrant (matching reference collection format)
        # Only include 'question' and 'question_id' fields
        qdrant_payload = {
            'question': payload.get('question', ''),
            'question_id': payload.get('question_id', '')
        }

        # Use numeric index as point ID (Qdrant requires unsigned int or UUID)
        point_id = idx

        # Create PointStruct
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=qdrant_payload
        )

        points.append(point)

    print(f"Prepared {len(points)} points for upload (skipped {skipped} without embeddings)")

    # Upload in batches
    total_uploaded = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            total_uploaded += len(batch)
            print(f"Uploaded batch {i // batch_size + 1}: {total_uploaded}/{len(points)} points")
        except Exception as e:
            print(f"Error uploading batch {i // batch_size + 1}: {e}")
            # Continue with next batch even if one fails

    print(f"\nUpload completed!")
    print(f"Total points uploaded: {total_uploaded}")
    print(f"Total points skipped: {skipped}")

    return total_uploaded

if __name__ == "__main__":
    """
    Main execution block. Uncomment the function you want to run.
    All configuration is set at the top of the file in the CONFIGURATION section.
    """

    # ========================================================================
    # Option 1: Display existing records from Qdrant collection
    # ========================================================================
    # connect_and_display_records()

    # ========================================================================
    # Option 2: Create a new Qdrant collection
    # ========================================================================
    # create_bdc_collection()

    # ========================================================================
    # Option 3: Generate questions using LLM (full pipeline)
    # ========================================================================
    # run_question_generation_pipeline()

    # ========================================================================
    # Option 4: Test question generation with limited data
    # ========================================================================
    # test_question_generation_pipeline()

    # ========================================================================
    # Option 5: Generate embeddings for questions
    # ========================================================================
    # generate_embeddings()

    # ========================================================================
    # Option 6: Upload embeddings to Qdrant
    # ========================================================================
    upload_embeddings_to_qdrant()