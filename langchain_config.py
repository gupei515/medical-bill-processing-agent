import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain configuration
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Verify LangChain API key is loaded
if not os.getenv('LANGCHAIN_API_KEY'):
    raise ValueError("LangChain API key not found. Please check your .env file.") 