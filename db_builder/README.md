# Qdrant Question Generation Pipeline

Pipeline for generating natural language questions from study descriptions using LLM, creating vector embeddings, and uploading to Qdrant vector database.

## Setup

1. **Install dependencies:**
   ```bash
   pip install qdrant-client langchain-ollama pandas requests
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Run the pipeline:**
   ```bash
   python qdrant_loader.py
   ```

## Configuration

All configuration is managed through environment variables in the `.env` file.

| Variable | Description | Default |
|----------|-------------|---------|
| **File Paths** |||
| `INPUT_STUDIES_FILE` | Path to studies JSON file | `99_select_studies.json` |
| `INPUT_PERSONAS_FILE` | Path to personas CSV file | `personas.csv` |
| `OUTPUT_QUESTIONS_FILE` | Generated questions output | `generated_questions_payloads.json` |
| `INPUT_EMBEDDINGS_FILE` | Questions with embeddings | `generated_questions_payloads_with_embeddings.json` |
| **LLM** |||
| `LLM_BASE_URL` | LLM service endpoint | `http://localhost:52236` |
| `LLM_MODEL_NAME` | Model for question generation | `google/gemma-3-12b-it` |
| **Embeddings** |||
| `EMBEDDING_BASE_URL` | Embedding service endpoint | `http://localhost:11434` |
| `EMBEDDING_MODEL_NAME` | Embedding model | `bge-m3` |
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding | `100` |
| `EMBEDDING_SAVE_INTERVAL` | Save interval (embeddings) | `50` |
| **Qdrant** |||
| `QDRANT_URL` | Qdrant server URL | `http://localhost:60850` |
| `QDRANT_COLLECTION_NAME` | Target collection name | `Collection_name` |
| `VECTOR_SIZE` | Embedding dimensions | `1024` |
| `DISTANCE_METRIC` | Distance metric | `Cosine` |
| `UPLOAD_BATCH_SIZE` | Upload batch size | `100` |
| **Pipeline** |||
| `START_PERSONA` | Starting persona index | `1` |
| `MAX_STUDIES` | Limit studies (empty = all) | - |
| `MAX_PERSONAS` | Limit personas (empty = all) | - |
| `SAVE_INTERVAL` | Save interval (questions) | `100` |

## Usage

The pipeline provides six operational modes. Edit the `if __name__ == "__main__":` block in `qdrant_loader.py` to uncomment your desired option.

### Available Operations

| Option | Function | Description |
|--------|----------|-------------|
| 1 | `connect_and_display_records()` | View sample records from Qdrant collection |
| 2 | `create_bdc_collection()` | Create new Qdrant collection with config settings |
| 3 | `run_question_generation_pipeline()` | Generate questions from studies (full dataset) |
| 4 | `test_question_generation_pipeline()` | Test generation with limited data (5 studies, 1 persona) |
| 5 | `generate_embeddings()` | Create vector embeddings for questions |
| 6 | `upload_embeddings_to_qdrant()` | Upload embeddings to Qdrant |

### Complete Workflow

For a full end-to-end execution:

```bash
# Step 1: Generate questions from studies
# Uncomment: run_question_generation_pipeline()
python qdrant_loader.py

# Step 2: Generate embeddings
# Uncomment: generate_embeddings()
python qdrant_loader.py

# Step 3: Upload to Qdrant
# Uncomment: upload_embeddings_to_qdrant()
python qdrant_loader.py
```

## Pipeline Flow

```
Studies (JSON) → Question Generation (LLM) → Embedding Generation (Ollama) → Upload to Qdrant
```

## Notes

- Progress is saved incrementally to avoid data loss
- Use `START_PERSONA` parameter to resume interrupted runs
- All configuration managed through `.env` file
- Collection is created automatically if it doesn't exist
