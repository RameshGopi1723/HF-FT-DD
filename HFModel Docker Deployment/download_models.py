from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def download_model(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

# Download English-French and French-English models
download_model('models/en_fr/', 'Helsinki-NLP/opus-mt-en-fr')
download_model('models/fr_en/', 'Helsinki-NLP/opus-mt-fr-en')