from langchain.schema import Document
from langchain.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import get_embedding_model, get_db_collection


def create_chunk_embeddings(document_pages: list):
    """
    Tool to create embeddings for text chunks extracted from PDF pages using RecursiveCharacterTextSplitter.
    Args:
        document_pages (list): List of PDF pages as Document objects.

    This function uses the SentenceTransformer model to create embeddings for text chunks
    extracted from PDF pages. It splits the text into manageable chunks using RecursiveCharacterTextSplitter
    and then encodes these chunks into embeddings. The embeddings are returned as a list.
    """

    model = get_embedding_model()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for i, page in enumerate(document_pages):
        text = page.extract_text()
        if text:
            chunks.extend(splitter.split_text(text))

    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)

    return embeddings

chunk_embedding_tool = Tool(
    name="create_chunk_embeddings",
    func=create_chunk_embeddings,
    description="Create embeddings for text chunks extracted from PDF pages using RecursiveCharacterTextSplitter. "
)


def find_matching_policies(query: str, top_k: int=3):
    """
    Tool to find matching policies based on a query using embeddings.
    Args:
        query (str): The query to find matching policies for.
        top_k (int): The number of top matching policies to return.
    """

    model = get_embedding_model()

    # Get the policy collection from the database
    policy_collection = get_db_collection("policies")

    q_embedding = model.encode([query])[0]
    results = policy_collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results

matching_policy_tool = Tool(
    name="find_matching_policies",
    func=find_matching_policies,
    description="Find matching policies based on a query using embeddings. Returns top K matching policies with their documents and metadata."
)


def find_similar_documents(query: str, top_k: int=3):
    """
    Tool to find similar documents based on a query and policies to give more context to back up answer.
    
    Args:
        query (str): The query to find similar documents for.
        top_k (int): The number of top similar documents to return.
    """
    
    model = get_embedding_model()

    # Get the document collection from the database
    document_collection = get_db_collection("enterprise_docs")

    q_embedding = model.encode([query])[0]
    results = document_collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results


similar_document_tool = Tool(
    name="find_similar_documents",
    func=find_similar_documents,
    description="Find similar documents based on a query and policies to give more context to back up answer. Returns top K similar documents with their metadata."
)
