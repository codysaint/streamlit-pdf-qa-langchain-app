import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
import os


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    else:
        print(f" {directory_path} already exists")

#llm 
llm_name = "gpt-3.5-turbo"

# persist_directory
persist_directory = 'vector_index/'

create_directory_if_not_exists(persist_directory)

docs_dir = 'mydocs/'

docs_sqlite_store_chroma = os.path.join(persist_directory, "chroma_vec_store.sqlite3")

# @st.cache_resource
def chain_workflow(openai_api_key):

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    

    # Check if the file exists
    if not os.path.exists(docs_sqlite_store_chroma):
        # If it doesn't exist, create it

        # load multiple pdfs at once
        # loader = PyPDFDirectoryLoader(docs_dir)

        # load single document
        file = os.path.join(docs_dir, "key_highlights.pdf")
        loader = PyPDFLoader(file)

        documents = loader.load()

        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        # persist_directory
        # persist_directory = 'vector_index/'

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )


        vectordb.persist()
        print(f"Vectorstore created and saved successfully, The {docs_sqlite_store_chroma} file has been created.")
    else:
        # if vectorstore already exist, just call it
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    
    # specify a retrieval to retrieve relevant splits or documents
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}))

    
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=5,memory_key="chat_history")
    
    # create a chatbot chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0.7, openai_api_key=openai_api_key), 
        chain_type="stuff", 
        retriever=compression_retriever, 
        memory=memory,
        get_chat_history=lambda h : h,
        verbose=True
    )
    
    
    return qa