import os
from dotenv import load_dotenv
load_dotenv()


LLM_MODE = os.getenv("LLM_MODE", "local")  # "local" or "api"

# For API mode
API_KEY = os.getenv("API_KEY", "")
API_MODEL = os.getenv("API_MODEL", "llama3-8b-8192")