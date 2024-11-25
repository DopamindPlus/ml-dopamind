from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# Inisialisasi Flask
app = Flask(__name__)

# Parameter
max_length = 128

# Lazy Loading
model = None
tokenizer = None
label_encoder = None

def load_resources():
    global model, tokenizer, label_encoder
    if not model or not tokenizer or not label_encoder:
        # Muat tokenizer
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-lite-large-p2')

        # Muat label encoder
        with open('label_encoder/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Muat model
        bert_model = TFAutoModel.from_pretrained('indobenchmark/indobert-lite-large-p2')
        input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
        attention_masks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_masks')
        pooled_output = bert_model(input_ids, attention_mask=attention_masks).pooler_output
        dropout = tf.keras.layers.Dropout(0.3)(pooled_output)
        output = tf.keras.layers.Dense(4, activation='softmax')(dropout)
        model_temp = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
        model_temp.load_weights('model_weights/nlp_emotion_indobert.h5')
        model = model_temp

def preprocess_texts(texts):
    inputs = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="np"
    )
    return inputs['input_ids'], inputs['attention_mask']

@app.route('/predict', methods=['POST'])
def predict():
    # Muat resource jika belum dimuat
    load_resources()

    # Ambil data teks dari request
    data = request.get_json()
    texts = data.get('texts')

    # Validasi input
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        return jsonify({"error": "Input harus berupa string atau list"}), 400

    # Preproses teks
    input_ids, attention_masks = preprocess_texts(texts)

    # Prediksi
    predictions = model.predict({'input_ids': input_ids, 'attention_masks': attention_masks})
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    # Respons JSON
    return jsonify({
        'texts': texts,
        'predictions': predicted_labels.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
