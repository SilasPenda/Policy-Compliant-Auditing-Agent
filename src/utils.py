import os
import sys
import yaml
import torch
import chromadb
import platform
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI


load_dotenv()

# Get the absolute path to the root of the project
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add src to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["OPENAI_API_KEY"]
openai_api_key = os.getenv("OPENAI_API_KEY")


def read_pdf(file_path):
    """
    Reads a PDF file and returns its content as a list of strings, each representing a page.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of pages, where each page is represented as a string.
    """

    try:
        document = PdfReader(file_path)
        return document.pages
    
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return []


def read_yaml(file_path):
    """
    Reads a YAML file and returns the content as a Python dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """

    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        return content
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None
    

def get_device():
    """
    Get the device to be used for tensor operations.
    Returns:
        torch.device: The device to be used (CPU, CUDA, or MPS).
    """

    if platform.system() == "Darwin":
        if torch.backends.mps.is_available():
            return "mps"
        
    else:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
        

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Get a pre-trained embedding model based on the specified model name.
    
    Args:
        model_name (str): The name of the embedding model to use.
        
    Returns:
        SentenceTransformer: An instance of the specified embedding model.
    """

    device = get_device()
    return SentenceTransformer(model_name, device=device)


def get_llm(type="openai", model_name="gpt-4o"):
    """
    Get a language model based on the specified type and model name.
    Args:
        type (str): The type of language model to use ("openai" or "ollama").
        model_name (str): The name of the model to use.
    Returns:
        llm: An instance of the specified language model.
    """

    llms = {
        "openai": ChatOpenAI(model_name=model_name, openai_api_key = openai_api_key),
        "ollama": OllamaLLM(model=model_name)
    }
    llm = llms[type]

    return llm


def get_db_collection(collection_name, persist_dir = "./audit_chromadb_dir"):
    """
    Get a collection from the ChromaDB client.

    Args:
        collection_name (str): The name of the collection to retrieve.

    Returns:
        chromadb.Collection: The requested collection.
    """

    try:
        os.makedirs(persist_dir, exist_ok=True) 
        client = chromadb.PersistentClient(path=persist_dir)
        return client.get_collection(name=collection_name)
    
    except Exception as e:
        print(f"Error getting collection '{collection_name}': {e}")
        return None
