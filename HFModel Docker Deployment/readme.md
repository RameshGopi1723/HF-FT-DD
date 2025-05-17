# Deploying Hugging Face Models in a Docker Container

## Overview

This document explains how to deploy Hugging Face models in a Docker container and expose them as a translation web service using Flask. The service supports English-to-French and French-to-English translations.

## Prerequisites

* Docker installed (preferably Docker Desktop)
* Basic knowledge of Python and Docker

---

## Directory Structure

```plaintext
huggingface-docker-deploy/
├── Dockerfile
├── requirements.txt
├── main.py
├── download_models.py
```

---

## 1. Dockerfile (No Extension)

```dockerfile
# Use Ubuntu's current LTS
FROM ubuntu:jammy-20230804

# Install only necessary packages (Python, pip, venv)
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3 \
        python3-pip \
        python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY main.py .
COPY download_models.py .

# Create and activate virtual environment
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies and download models
RUN pip install --no-cache-dir -r requirements.txt && \
    python3 download_models.py

# Expose Flask service port
EXPOSE 6000

# Start the service
ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]
```

---

## 2. requirements.txt

```txt
transformers==4.30.2
flask==2.3.3
torch==2.0.1
sacremoses==0.0.53
sentencepiece==0.1.99
```

---

## 3. download\_models.py

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def download_model(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# Download models
download_model('models/en_fr/', 'Helsinki-NLP/opus-mt-en-fr')
download_model('models/fr_en/', 'Helsinki-NLP/opus-mt-fr-en')
```

---

## 4. main.py

```python
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load models and tokenizers
def get_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

en_fr_model, en_fr_tokenizer = get_model('models/en_fr/')
fr_en_model, fr_en_tokenizer = get_model('models/fr_en/')

app = Flask(__name__)

# Check if translation is supported
def is_translation_supported(from_lang, to_lang):
    return f'{from_lang}_{to_lang}' in ['en_fr', 'fr_en']

@app.route('/translate/<from_lang>/<to_lang>/', methods=['POST'])
def translate_endpoint(from_lang, to_lang):
    if not is_translation_supported(from_lang, to_lang):
        return jsonify({'error': 'Translation not supported'}), 400

    data = request.get_json()
    from_text = data.get(f'{from_lang}_text', '')

    if from_text:
        model, tokenizer = (en_fr_model, en_fr_tokenizer) if from_lang == 'en' else (fr_en_model, fr_en_tokenizer)
        to_text = tokenizer.decode(
            model.generate(tokenizer.encode(from_text, return_tensors='pt')).squeeze(),
            skip_special_tokens=True
        )
        return jsonify({f'{to_lang}_text': to_text})
    else:
        return jsonify({'error': 'Text to translate not provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
```

---

## 5. Build and Run the Container

### Build the Docker Image

```bash
docker build -t gopinath1723/hf-mode-deloyment-docker:latest .
```

### Run the Docker Container

```bash
docker run -p 6000:6000 gopinath1723/hf-mode-deloyment-docker:latest
```

---

## 6. Test the Translation Service

Use `curl` or Postman:

```bash
curl -X POST http://localhost:6000/translate/en/fr/ \
     -H "Content-Type: application/json" \
     -d '{"en_text": "Hello, how are you?"}'
```

Expected Output:

```json
{"fr_text": "Bonjour, comment ça va ?"}
```

---

## Conclusion

This setup allows you to deploy any Hugging Face model as a web service using Docker and Flask. It’s scalable, replicable, and great for testing ML models in isolated environments.
