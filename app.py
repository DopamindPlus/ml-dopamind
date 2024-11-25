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

# Lazy Loading untuk resource
tokenizer = None
label_encoder = None
model = None

def load_resources():
    global tokenizer, label_encoder, model
    if not tokenizer:
        # Muat tokenizer
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-lite-large-p2')

    if not label_encoder:
        # Muat label encoder dari file
        with open('label_encoder/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

    if not model:
        # Buat ulang model
        bert_model = TFAutoModel.from_pretrained('indobenchmark/indobert-lite-large-p2')

        input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
        attention_masks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_masks')

        # Dapatkan output dari model BERT
        sequence_output = bert_model(input_ids, attention_mask=attention_masks).last_hidden_state
        pooled_output = bert_model(input_ids, attention_mask=attention_masks).pooler_output

        # Gabungkan sequence output dan pooled output
        merged_output = tf.keras.layers.concatenate([sequence_output[:, 0, :], pooled_output])

        # Tambahkan lapisan tambahan untuk klasifikasi
        dense = tf.keras.layers.Dense(128, activation='relu')(merged_output)
        dropout = tf.keras.layers.Dropout(0.3)(dense)
        output = tf.keras.layers.Dense(4, activation='softmax')(dropout)

        model_temp = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)

        # Load bobot model yang telah dilatih
        model_temp.load_weights('model_weights/nlp_emotion_indobert.h5')
        model = model_temp

def preprocess_texts(texts, tokenizer, max_length):
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
    # Pastikan resource dimuat terlebih dahulu
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
    input_ids, attention_masks = preprocess_texts(texts, tokenizer, max_length)

    # Lakukan prediksi
    predictions = model.predict({'input_ids': input_ids, 'attention_masks': attention_masks})

    # Dapatkan probabilitas masing-masing kelas
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    # Buat respons JSON
    response = {
        'results': []
    }
    for i, text in enumerate(texts):
        response['results'].append({
            'text': text,
            'predicted_label': predicted_labels[i],
            'probabilities': {
                label: float(predictions[i][idx]) for idx, label in enumerate(label_encoder.classes_)
            }
        })
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
