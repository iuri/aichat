from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import os
import pickle

# Load knowledge base
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("embeddings.npy", "rb") as f:
    embeddings = np.load(f)

# Load model and FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Vonage Credentials (Set these as environment variables)
VONAGE_API_KEY = os.getenv("VONAGE_API_KEY")
VONAGE_API_SECRET = os.getenv("VONAGE_API_SECRET")
VONAGE_WHATSAPP_NUMBER = os.getenv("VONAGE_WHATSAPP_NUMBER")  # e.g., "14157386170"



app = Flask(__name__)

@app.route("/inbound", methods=["POST"])
def inbound():
    data = request.get_json()
    print("Inbound message received:", data)

    user_number = data['from']['number']
    message_text = data['message']['content']['text']

    # Step 1: Embed the user message
    q_embedding = model.encode([message_text])

    # Step 2: Search knowledge base
    D, I = index.search(np.array(q_embedding).astype("float32"), k=1)
    response_text = chunks[I[0][0]]

    # Step 3: Respond via Vonage Messages API
    send_whatsapp_message(user_number, response_text)

    return jsonify({"status": "ok"}), 200


@app.route("/status", methods=["POST"])
def status():
    data = request.get_json()
    print("Message status webhook:", data)
    return jsonify({"status": "received"}), 200


def send_whatsapp_message(to_number, text):
    url = "https://api.nexmo.com/v1/messages"

    payload = {
        "from": {
            "type": "whatsapp",
            "number": VONAGE_WHATSAPP_NUMBER
        },
        "to": {
            "type": "whatsapp",
            "number": to_number
        },
        "message": {
            "content": {
                "type": "text",
                "text": text
            }
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {os.getenv('VONAGE_AUTH_BASE64')}"  # Or use OAuth2 token if preferred
    }

    response = requests.post(url, json=payload, headers=headers)
    print("Sent message:", response.json())





@app.route("/", methods=["GET", "POST"])
def home():
    return jsonify({"status": "ok"}), 200
    


    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
