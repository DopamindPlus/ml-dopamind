from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import logging
from logging.handlers import RotatingFileHandler
import jwt
import requests
import os

# Inisialisasi Flask
app = Flask(__name__)

port = int(os.environ.get("PORT", 8080))

# Menambahkan logging untuk produksi
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Konfigurasi JWT
SECRET_KEY = "CHANGETHISPLEASE" 

# Definisi tokenizer dan model
max_length = 128
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-lite-large-p2')

# Muat label encoder dari file
with open('label_encoder/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Buat ulang model
def create_model():
    bert_model = TFAutoModel.from_pretrained('indobenchmark/indobert-lite-large-p2')

    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_masks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_masks')

    # Dapatkan output dari model BERT
    sequence_output = bert_model(input_ids, attention_mask=attention_masks).last_hidden_state
    pooled_output = bert_model(input_ids, attention_mask=attention_masks).pooler_output

    # Gabungkan sequence output dan pooled output
    merged_output = tf.keras.layers.concatenate([sequence_output[:, 0, :], pooled_output])

    # Menambahkan lapisan tambahan untuk klasifikasi
    dense = tf.keras.layers.Dense(128, activation='relu')(merged_output)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(4, activation='softmax')(dropout)

    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
    return model

model = create_model()

# Load bobot model yang telah dilatih
model.load_weights('model_weights/nlp_emotion_indobert.h5')  # Path bobot model Anda

# Middleware untuk verifikasi token
def auth_middleware(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Token tidak ditemukan"}), 403

        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = decoded 
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token telah kadaluarsa"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Token tidak valid"}), 401

        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# Fungsi untuk memproses teks
def preprocess_texts(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_masks)

@app.route('/predict', methods=['POST'])
@auth_middleware
def predict():
    try:
        # Ambil data teks dari request
        data = request.get_json()
        texts = data.get('texts')  # Ambil key 'texts'

        # Jika input adalah string, ubah menjadi list
        if isinstance(texts, str):
            texts = [texts]

        # Jika input bukan string atau list, kembalikan error
        if not isinstance(texts, list):
            return jsonify({"error": "Input harus berupa string atau list"}), 400

        # Preproses teks
        input_ids, attention_masks = preprocess_texts(texts, tokenizer, max_length)

        # Lakukan prediksi
        predictions = model.predict({'input_ids': input_ids, 'attention_masks': attention_masks})
        predicted_classes = np.argmax(predictions, axis=1)

        # Konversi indeks ke label deskriptif
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        # Buat respons JSON
        response = {
            'texts': texts,
            'predictions': predicted_labels.tolist()
        }

        # Kirim data ke backend untuk disimpan di database
        save_response = {
            "texts": texts,
            "predictions": predicted_labels.tolist(),
        }
        headers = {
            "Authorization": request.headers.get('Authorization'), 
            "Content-Type": "application/json"
        }
        backend_url = "http://34.101.142.68:9000/api/mood"
        save_result = requests.post(backend_url, json=save_response, headers=headers)

        if save_result.status_code != 201:
            app.logger.error("Failed to save prediction to database")
            return jsonify({"error": "Gagal menyimpan prediksi ke database"}), 500

        return jsonify(response)

    except Exception as e:
        # Log error jika ada kesalahan dalam proses prediksi
        app.logger.error(f"Error processing prediction: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan saat memproses permintaan"}), 500

# Menjalankan aplikasi dengan Gunicorn
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
