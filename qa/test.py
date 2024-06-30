from dotenv import load_dotenv
load_dotenv()
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import tempfile
import os
import streamlit as st
from langchain.document_loaders import TextLoader

from semantic_router import Route
from semantic_router.encoders import  OpenAIEncoder
from semantic_router.layer import RouteLayer

def textfile_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = TextLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

def uploadFileToDB(uploaded_file):
    

    #업로드 되면 동작하는 코드
    if uploaded_file is not None:
        # pages = pdf_to_document(uploaded_file)
        pages = textfile_to_document(uploaded_file)

        text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        )

        texts = text_splitter.split_documents(pages)

        embeddings_model = OpenAIEmbeddings()

        #db = Chroma.from_documents(texts, embeddings_model, persist_directory="/chroma")
        db = Chroma.from_documents(texts, embeddings_model)
        return db
    # else:
    #     print('파일이 존재하지 않아 기존 DB 활용')
        # return Chroma.from_documents(texts, embeddings_model, persist_directory="/chroma")
    


uploaded_file = st.file_uploader("로그 파일을 업로드하세요")
st.write("---")
db = uploadFileToDB(uploaded_file)

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm)

question = 'trans로그를 추가하려는데 코드 작성 가이드를 알려줘'
docs = retriever_from_llm.get_relevant_documents(query=question)
print(docs)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
result = qa_chain({"query":question})
st.write(result["result"])


# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# from langchain.agents import AgentType, initialize_agent
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferWindowMemory


# memory1 = ConversationBufferWindowMemory(
#     memory_key="chat_history", k=5, return_messages=True, output_key="output"
# )
# memory2 = ConversationBufferWindowMemory(
#     memory_key="chat_history", k=5, return_messages=True, output_key="output"
# )

# agent = initialize_agent(
#     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     tools=[],
#     llm=llm,
#     max_iterations=3,
#     early_stopping_method="generate",
#     memory=memory1,
# )

# new_prompt = agent.agent.create_prompt(system_message=system_message, tools=[])
# agent.agent.llm_chain.prompt = new_prompt

# print("agent")
# result = agent(input_query)
# print(result)

# agent.memory = memory2
# agent()






