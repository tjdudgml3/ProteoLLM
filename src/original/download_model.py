import os
import requests

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json"
]

DEST_DIR = "models/all-MiniLM-L6-v2"
os.makedirs(DEST_DIR, exist_ok=True)

BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"

for file in FILES:
    url = f"{BASE_URL}/{file}"
    dest_path = os.path.join(DEST_DIR, file)
    print(f"Downloading {file} from {url}...")
    
    try:
        response = requests.get(url, verify=False, stream=True)
        response.raise_for_status()
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {dest_path}")
    except Exception as e:
        print(f"Failed to download {file}: {e}")
