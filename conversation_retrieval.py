from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from recursive_text_splitter import get_documents_from_web
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder



#instantiate model

llm = ChatOpenAI(model = "gpt-3.5-turbo")

def get_documents_from_web(url):
    #load document
    loader = WebBaseLoader(url)
    docs = loader.load()
    #split docs in smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 20
    )
    splitdocs = splitter.split_documents(docs)
    return splitdocs


def createVecorDB(doc):
    embedding = OpenAIEmbeddings()
    #convert docs into vector
    vectorStore = FAISS.from_documents(doc, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    # create prompt
    #prompt = ChatPromptTemplate.from_template("""Answer user question = {input} context= {context}""")
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer user question based on the: {context}"),
        MessagesPlaceholder(variable="chat_history"),
        ("human","{input}")])

    #create chain
    chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
    #create retrieval chain
    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    

    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain


def process_chat(chain, user_input, chat_history):
    response = chain.invoke({
        "input":user_input,
        "chat_history":chat_history})
    return(response['answer'])

if __name__=='__main__':
    docs = get_documents_from_web('https://www.yosemite.com/things-to-do/half-dome-hike/')
    vectorStore = createVecorDB(docs)
    chain = create_chain(vectorStore)

    chat_history=[]

    while True:
        user_input = input("You: ")
        if user_input.lower()=="exit":
            break
        response = process_chat(chain, user_input,chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant",response)
        

