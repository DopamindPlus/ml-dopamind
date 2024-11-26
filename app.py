from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# Inisialisasi Flask
app = Flask(__name__)

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
def predict():
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
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
