# ML-Dopamind 🧠💬

## Project Overview
ML-Dopamind is a machine learning project focused on emotion detection using Natural Language Processing (NLP). The project aims to classify emotional states in text conversations.

## Emotion Classification Categories
- Happiness
- Sadness
- Fear
- Anger

## Repository Structure
### Branches
* **main**: Primary development branch
* **verry**: Individual development branch
* **richal**: Individual development branch
* **kiki**: Individual development branch

## Technical Specifications
### Model Architecture
- Framework: TensorFlow
- Pretrained Model: IndoBERT
- Tokenizer: BertTokenizer
- Classification: 4-class emotion detection

### Key Technologies
- Python
- TensorFlow
- Transformers (Hugging Face)
- Flask
- JWT Authentication

## Key Features
- Real-time emotion detection
- Multilingual support (Indonesian)
- Secure API endpoint
- Logging and error handling

## API Endpoint
`/predict` 
- Method: POST
- Authentication: Bearer Token
- Input: Text or list of texts
- Output: Emotion prediction

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow
- Transformers
- Flask
- JWT

### Setup Steps
1. Clone the repository
2. Install dependencies
3. Download pre-trained weights
4. Configure environment variables

## Usage Example
```python
# Sample prediction request
{
    "texts": ["Aku sedang merasa senang hari ini"],
    "Authorization": "Bearer <your_token>"
}
```

## Security Features
- JWT Token Authentication
- Input validation
- Error logging
- Secure model inference

## Deployment
- Gunicorn server
- Configurable port
- Cloud-ready configuration

## Development Team
- Verry
- Richal
- Kiki

## License
[Insert Appropriate License]

## Disclaimer
Experimental emotion detection system. Not a substitute for professional psychological assessment.

## Model

Link model nlp_emotion_indobert.h5 : https://drive.google.com/drive/folders/1-ngrmQ4lJyGrkOp2UWqhx2qeITodD4Za
