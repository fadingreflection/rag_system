import os

from dotenv import load_dotenv

# Загружаем переменные из .env (если файл существует)
load_dotenv()

# API
# APP_NAME = os.getenv("APP_NAME", "RAG Service")
# DEBUG = os.getenv("DEBUG", "False").lower() == "true"
HOST = os.getenv("UVICORN_HOST", "0.0.0.0")
PORT = int(os.getenv("UVICORN_PORT", 8000))

# MODEL INFERENCE
DATA_ID = os.getenv("DATA_ID", "Den4ikAI/russian_cleared_wikipedia")
SPLITTER = os.getenv("SPLITTER", "train")
RECORD_KEY = os.getenv("RECORD_KEY", "sample")
MODEL_ID_SS = os.getenv("MODEL_ID_SS", "intfloat/multilingual-e5-base")
DB_DIR = os.getenv("DB_DIR", "chroma_ragmini")
MODEL_ID = os.getenv("MODEL_ID", "unsloth/Qwen2.5-3B-unsloth-bnb-4bit" )# неплохо работает на русском, нужный размер эмбеддингов 768
TASK_TYPE = os.getenv("TASK_TYPE", "text-generation")
MARKER = os.getenv("MARKER", "</think>")
USE_CUSTOM_DATA_FLAG = os.getenv("USE_CUSTOM_DATA_FLAG", True)
RECREATE_DB_FLAG = os.getenv("RECREATE_DB_FLAG", True)

# NLP PIPE: PROMPTING
MAX_NEW_TOKENS=os.getenv("MAX_NEW_TOKENS", 256)
DO_SAMPLE=os.getenv("DO_SAMPLE", True) #позволяем модели додумывать
TRUNCATION=os.getenv("truncation", True)
TOP_K = os.getenv("TOP_K", 5)
NUM_RETURN_SEQUENCES = os.getenv("NUM_RETURN_SEQUENCES", 1)
SEED = os.getenv("SEED", 42)
TEMPERATURE = os.getenv("TEMPERATURE", 0)

# RAG CONSTRUCT
NUM_SEARCH_CHUNKS = os.getenv("NUM_SEARCH_CHUNKS", 5)

# DB BUILDER
MODEL_ID_SS = os.getenv("MODEL_ID_SS", "intfloat/multilingual-e5-base")
DB_DIR = os.getenv("DB_DIR", "chroma_ragmini")
DB_DIR_CUSTOM = os.getenv("DB_DIR_CUSTOM", "my_personal_docs_db")
CUSTOM_DOC_PATH = os.getenv("CUSTOM_DOC_PATH", r"C:\Users\Anastasiya Fedotova\Desktop\avtoreferat_psl_FEDOTOVA_s_uchetom_zamechaniy.docx")
LOAD_VIA_LANGCHAIN_FLAG = os.getenv("LOAD_VIA_LANGCHAIN_FLAG", False)
CHUNK_SIZE=os.getenv("CHUNK_SIZE", 256)
CHUNK_OVERLAP=os.getenv("CHUNK_OVERLAP", 64)

# # === Другие глобальные константы ===
# DATA_DIR = os.getenv("DATA_DIR", "data")
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

