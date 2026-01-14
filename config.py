import os
import ssl
import google.genai as genai
from google.genai import types

# --- AGGRESSIVE SSL PATCH ---
# This must be done before any other imports that use SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
# ---------------------------

# API Key
GOOGLE_API_KEY = ""
# Try to get from env
if "GOOGLE_API_KEY" in os.environ:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Modelsgemini-2.5-flash
FAST_MODEL = "gemini-2.5-flash"
# FAST_MODEL = "gemini-2.0-flash"
SMART_MODEL = "gemini-3-pro-preview" # Fallback to Flash as Pro Exp not found.

# User requested constants
MODEL_FAST = FAST_MODEL
MODEL_HIGH_REASONING = SMART_MODEL
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1

# Paths
DATA_DIR = os.path.dirname(os.path.dirname(__file__))
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db_index")

# Patch requests
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
session.verify = False

_orig_request = requests.request
def patched_request(*args, **kwargs):
    kwargs['verify'] = False
    return _orig_request(*args, **kwargs)
requests.request = patched_request

# Patch httpx (used by ADK)
import httpx
original_init = httpx.AsyncClient.__init__
def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = patched_init
