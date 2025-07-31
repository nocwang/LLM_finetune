# Text Tagging Model Fine-Tuning and API

This project provides a framework for fine-tuning a language model to generate tags for text inputs and deploying a FastAPI backend to serve predictions. It is designed to be flexible for various text tagging tasks, such as categorizing titles or descriptions, and is deployed using Docker for scalability.

## Overview

- **finetune.py**: Fine-tunes a pre-trained language model (Mistral-7B-Instruct-v0.2-GPTQ) using LoRA for efficient training on a text tagging dataset.
- **backend.py**: A FastAPI service to serve tag predictions using the fine-tuned model.
- **Dockerfile**: Builds a Docker image for the FastAPI backend.
- **.env**: Stores configuration parameters (e.g., model name, Hugging Face token).
- **requirements.txt**: Lists Python dependencies for the backend.

## Prerequisites

- Python 3.10+
- Docker (for deployment)
- Hugging Face account with a valid token
- CUDA-capable GPU (recommended for training and inference)
- Dataset in CSV format with input text and tags (e.g., `title` and `tags` columns)

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nocwang/LLM_finetune.git
cd LLM_finetune
```

### 2. Configure Environment Variables
Create a `.env` file in the project root with the variables:

Replace `your_huggingface_token` with your Hugging Face token and update `DATA_PATH`, `INPUT_COLUMN`, and `TARGET_COLUMN` to match your dataset.

### 3. Install Dependencies
For fine-tuning, install the required packages:
```bash
pip install -r requirements.txt
pip install accelerate peft bitsandbytes git+https://github.com/huggingface/transformers trl py7zr auto-gptq optimum datasets
```

### 4. Prepare Dataset
Ensure your dataset is a CSV file with at least two columns: one for input text (e.g., `title`) and one for tags (e.g., `tags` as a list of strings). Tags should be provided as a comma-separated string or a list that can be parsed with `ast.literal_eval`.

Example dataset (`data.csv`):
```csv
title,tags
"Sample text for tagging","['Tag1', 'Tag2', 'Tag3']"
"Another text example","['Category1', 'Category2', 'Category3', 'Category4']"
```

## Fine-Tuning the Model

Run the fine-tuning script to train the model on your dataset:
```bash
python finetune.py
```

- The script loads the dataset, processes it into prompts, and fine-tunes the model using LoRA.
- The fine-tuned model is saved to the `OUTPUT_DIR` and pushed to the Hugging Face Hub (requires `HF_TOKEN`).

## Deploying the API

### 1. Build the Docker Image
```bash
docker build -t tag-predictor .
```

### 2. Run the Docker Container
```bash
docker run -p 8000:8000 --env-file .env tag-predictor
```

The API will be available at `http://localhost:8000`.

### 3. Test the API
Use a tool like `curl` or Postman to send a POST request to the `/predict` endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Sample text for tagging", "Another text example"]}'
```

**Response**:
```json
[
  {"tags": "['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']"},
  {"tags": "['Category1', 'Category2', 'Category3', 'Category4', 'Category5']"}
]
```

Check the API health:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{"status": "healthy"}
```

## Usage Notes

- **Generalization**: The code is designed to handle any text tagging task. Update the `.env` file to point to your dataset and adjust column names as needed.
- **Model**: The default model is `Mistral-7B-Instruct-v0.2-GPTQ`. Change `MODEL_NAME` in `.env` to use a different model.
- **Performance**: Adjust `LORA_R`, `MAX_SEQ_LENGTH`, and `MAX_NEW_TOKENS` in `.env` to balance performance and resource usage.
- **Dataset**: Ensure tags are relevant and in the same language as the input text. The model expects up to five tags per input.

## Troubleshooting

- **Hugging Face Authentication**: Ensure `HF_TOKEN` is valid and has write permissions for pushing models.
- **GPU Memory**: If you encounter OOM errors, reduce `BATCH_SIZE` or `MAX_SEQ_LENGTH` in `.env`.
- **Docker Issues**: Verify the `.env` file is correctly loaded into the container using `--env-file`.

## License

This project is licensed under the following terms:

- **Academic Use**: Free to use for non-commercial academic research, provided that you cite the following repository in any publications or presentations:
  ```
  Wang, T. (2025). LLM_finetune: A framework for fine-tuning language models for text tagging. GitHub Repository, https://github.com/nocwang/LLM_finetune
  ```
- **Commercial Use**: Requires a paid license. Please contact the repository owner to inquire about commercial licensing terms.
