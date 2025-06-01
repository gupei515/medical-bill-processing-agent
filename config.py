import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Verify API key is loaded
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please check your .env file.") 