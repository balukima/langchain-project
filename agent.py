import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent,AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from recursive_text_splitter import get_documents_from_web
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langsmith import Client
from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.smith import RunEvalConfig, run_on_dataset

load_dotenv(find_dotenv())
os.environ["TAVILY_API_KEY"]='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'
os.environ["LANGCHAIN_API_KEY"]='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'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Agent.py"


loader = WebBaseLoader("https://www.travelandleisure.com/trip-ideas/adventure-travel/how-to-travel-to-machu-picchu")
docs = loader.load()
#split docs in smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 20
)
splitdocs = splitter.split_documents(docs)
embedding = OpenAIEmbeddings()
#convert docs into vector
vectorStore = FAISS.from_documents(splitdocs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k":3})

llm = ChatOpenAI(
    model = "gpt-3.5-turbo-1106",
    temperature = 0.7
)
prompt = ChatPromptTemplate.from_messages(
    [("system","You are a friendly assistant called Max"),
     MessagesPlaceholder(variable_name="chat_history"),
     ("human","{input}"),

     MessagesPlaceholder(variable_name="agent_scratchpad")]
)

retriever_tools = create_retriever_tool(
    retriever,
    'peru',
    "Use this tool to find information about Machu Pichu")
search = TavilySearchResults()
tools = [search, retriever_tools]

agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input,chat_history):
    response = agentExecutor.invoke({
        "input":user_input,
        "chat_history":chat_history
    })
    return response["output"]

if __name__=='__main__':
    chat_history=[]

    while True:
        user_input = input("You: ")
        if user_input.lower()=="exit":
            break
        response = process_chat(agentExecutor, user_input,chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant",response)