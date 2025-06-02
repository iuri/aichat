from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
from requests.auth import HTTPBasicAuth
import os
import pickle

from dotenv import load_dotenv

import subprocess
import json

# Load knowledge base
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("embeddings.npy", "rb") as f:
    embeddings = np.load(f)

# Load model and FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

load_dotenv()
# Vonage Credentials (Set these as environment variables)
VONAGE_API_KEY = os.getenv("VONAGE_API_KEY")
VONAGE_API_SECRET = os.getenv("VONAGE_API_SECRET")
VONAGE_WHATSAPP_NUMBER = os.getenv("VONAGE_WHATSAPP_NUMBER")  # e.g., "14157386170"


app = Flask(__name__)

@app.route("/inbound", methods=["POST"])
def inbound():
    data = request.get_json()
    print("Inbound message received:", data)

    user_number = data['from']
    message_text = data['text']

    # Step 1: Embed the user message
    q_embedding = model.encode([message_text])

    # Step 2: Search knowledge base
    D, I = index.search(np.array(q_embedding).astype("float32"), k=1)
    response_text = chunks[I[0][0]]
    
    if not response_text:
        if data['text'] == 'Join dried joy':
            response_text = 'Welcome to AMX AI Chat Support! How could we assist you?'
        else:
            response_text = "It seems you got really interested on AMX!. Perhaps it'd be a good idea to contact us for further information about us! Call at +1 727-250-5661 \n\n May we help you with anything else?"

    print("RESPONSE", response_text)
    
    # Step 3: Respond via Vonage Messages API       
    # send_whatsapp_message(user_number, response_text)
    run_curl(user_number, response_text)

    return jsonify({"status": "ok"}), 200


@app.route("/status", methods=["POST"])
def status():
    data = request.get_json()
    print("Message status webhook:", data)
    return jsonify({"status": "received"}), 200




def run_curl(to_number, text):


    data = json.dumps({
        "from": "14157386102",
        "to": str(to_number),
        "message_type": "text",
        "text": text,
        "channel": "whatsapp"
        })

    # The actual curl command as a list of arguments
    curl_command = [
        "curl",
        "-X", "POST", "https://messages-sandbox.nexmo.com/v1/messages",
        "-u", " 4d4e06f2:7S2bJ36ZFjOP8xXS",        
        "-H", "Content-Type: application/json",
        "-H", "Accept: application/json",
        "-d", data
    ]

    
    # Run the curl command and capture output
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    # Print the output and errors
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    
    
def send_whatsapp_message(to_number, text):
   
    url = "https://messages-sandbox.nexmo.com/v1/messages"

    payload = {
        "from": VONAGE_WHATSAPP_NUMBER,
        "to": to_number,
        "message_type": "text",
        "text": text,
        "channel": "whatsapp"}


    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    auth = HTTPBasicAuth(VONAGE_API_KEY,VONAGE_API_SECRET)
    response = requests.post(url, json=payload, headers=headers, auth=auth)
    print("Sent message:", response.json())

    return


@app.route("/", methods=["GET", "POST"])
def home():
    return jsonify({"status": "ok"}), 200
    


    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
