from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from transformers import AlbertTokenizer, TFAutoModel, AutoTokenizer
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Path model dan tokenizer
MODEL_PATH = "nlp_emotion_indobert.h5"
TOKENIZER_NAME = "indobenchmark/indobert-lite-large-p2"

# Daftar label sesuai model Anda (sesuaikan dengan urutan output model)
labels_translated = ["Kegembiraan", "Kemarahan", "Kesedihan", "Ketakutan"]

# Softmax manual menggunakan NumPy
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Stabilkan dengan pengurangan max
    return exp_logits / np.sum(exp_logits)

# Load Model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None

# Load Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat tokenizer: {e}")
    tokenizer = None


@app.route("/")
def home():
    return "NLP Emotion API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Validasi input
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Input must contain 'text' key"}), 400

    input_text = data["text"]

    try:
        # Pastikan tokenizer dan model telah dimuat
        if tokenizer is None or model is None:
            return jsonify({"error": "Model or tokenizer failed to load"}), 500

        # Tokenize teks
        tokens = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=128,  # Sesuaikan dengan konfigurasi model Anda
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors="tf"
        )

        # Debugging: Cek tokenisasi
        print(f"Tokens: {tokens}")

        # Prediksi
        prediction = model.predict({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})

        # Normalisasi skor dengan softmax manual
        emotion_scores = softmax(prediction[0])
        predicted_label = np.argmax(emotion_scores)

        # Debugging: Cek hasil prediksi
        print(f"Prediction Scores: {emotion_scores}")

        # Ambil label emosi berdasarkan prediksi
        predicted_emotion = labels_translated[predicted_label]

        return jsonify({
            "text": input_text,
            "predicted_label": int(predicted_label),
            "predicted_emotion": predicted_emotion,
            "scores": emotion_scores.tolist()
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
