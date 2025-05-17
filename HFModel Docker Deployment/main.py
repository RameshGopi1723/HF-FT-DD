from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

# Load models once
en_fr_model, en_fr_tokenizer = get_model('models/en_fr/')
fr_en_model, fr_en_tokenizer = get_model('models/fr_en/')

app = Flask(__name__)

def is_translation_supported(from_lang, to_lang):
    return f'{from_lang}_{to_lang}' in ['en_fr', 'fr_en']

@app.route('/translate/<from_lang>/<to_lang>/', methods=['POST'])
def translate_endpoint(from_lang, to_lang):
    if not is_translation_supported(from_lang, to_lang):
        return jsonify({'error': 'Translation not supported'}), 400

    data = request.get_json()
    from_text = data.get(f'{from_lang}_text', '')

    if from_text:
        if from_lang == 'en':
            model, tokenizer = en_fr_model, en_fr_tokenizer
        elif from_lang == 'fr':
            model, tokenizer = fr_en_model, fr_en_tokenizer
        else:
            return jsonify({'error': 'Unsupported language'}), 400

        inputs = tokenizer.encode(from_text, return_tensors='pt')
        outputs = model.generate(inputs)
        to_text = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True)

        return jsonify({f'{to_lang}_text': to_text})

    return jsonify({'error': 'Text to translate not provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
