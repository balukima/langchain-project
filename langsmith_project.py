import os
from langsmith import Client
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.smith import RunEvalConfig, run_on_dataset

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"]='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'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langsmith_project"


client = Client()
llm = ChatOpenAI()
llm.invoke("Hello World")


