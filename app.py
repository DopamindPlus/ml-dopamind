from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel, TFAlbertModel, AlbertTokenizer
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Path model dan tokenizer
MODEL_PATH = "nlp_emotion_indobert.h5"
TOKENIZER_NAME = "bert-base-uncased"  # Ubah ke "albert-base-v2" jika menggunakan Albert    

# Daftar label sesuai model Anda (sesuaikan dengan urutan output model)
labels_translated = ["Kegembiraan", "Kemarahan", "Kesedihan", "Ketakutan"]

# Softmax manual menggunakan NumPy
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Stabilkan dengan pengurangan max
    return exp_logits / np.sum(exp_logits)

# Pemuatan model
try:
    # Pastikan lapisan sesuai dengan model Anda
    with tf.keras.utils.custom_object_scope({'TFBertModel': TFBertModel, 'TFAlbertModel': TFAlbertModel}):
        model = load_model(MODEL_PATH)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None

# Pemuatan tokenizer
try:
    # Gunakan tokenizer sesuai model
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)  # Ubah ke AlbertTokenizer jika Albert digunakan
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
        tokens = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=128,  # Sesuaikan dengan konfigurasi model Anda
            return_tensors="tf"
        )

        # Prediksi
        prediction = model.predict([tokens["input_ids"], tokens["attention_mask"]])

        # Normalisasi skor dengan softmax manual
        emotion_scores = softmax(prediction[0])
        predicted_label = np.argmax(emotion_scores)

        # Debugging (opsional, untuk memeriksa urutan label dan skor)
        print("Input Text:", input_text)
        print("Prediction Scores:", emotion_scores)
        for i, score in enumerate(emotion_scores):
            print(f"Label {i}: {score:.4f} -> {labels_translated[i]}")

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
