

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

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory


guide_q = Route(
    name="guide",
    utterances=[
        "trans로그를 추가하려는데 코드 작성 가이드를 알려줘",
        "slf4j 버전을 업그레이드 하려는데 패치 가이드를 알려줘",
        "datasource 접속정보 암호화를 하려는데 패치 가이드를 알려줘"
    ],
)

error_q = Route(
    name="error",
    utterances=[
        "R-CLIPS 오류 원인은?",
        "알클립스 오류 원인은?",
        "와치독 오류 원인은?",
        "WatchDog 오류 원인은?",
        "logstash 오류 error 원인은?",
        "로그스태시 오류 error 원인은?"
        "톰캣 오류 erorr 원인은?"
        "tomcat 오류 eror 원인은?",
        "filebeat 오류 error 원인은?",
        "파일비트 오류 error 원인은?"
        "jboss 오류 error 원인은?",
        "wildfly 오류 error 원인은?",
        "scouter 오류 error 원인은?",
        "h2o 오류 error 원인은?",
        "java 오류 error 원인은?"
    ]
)

routes = [error_q, guide_q]

# OpenAI
#os.environ["OPENAI_API_KEY"] = ""


# 테스트 웹페이지

# 사용자의 코드 설정파일 업로드
st.title("솔루션 코드 오류 문의")
st.write("---")


#Question
st.header("솔루션 문의하기")
question = st.text_input('문의 내용을 입력하세요')


# if st.button('질문하기'):
#     with st.spinner('처리중입니다.'):
#         llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
#         qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
#         result = qa_chain({"query":question})
#         st.write(result["result"])


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
    else:
        print('파일이 존재하지 않아 기존 DB 활용')
        return Chroma.from_documents(texts, embeddings_model, persist_directory="/chroma")



# 웹페이지 입력하여 가져올 경우 https://www.elastic.co/guide/en/logstash/current/keystore.html
# input_doc_url = st.text_input('필요시 documentation 링크를 입력하세요')
# # Load documents from web
# from langchain.document_loaders import WebBaseLoader

# web_loader = WebBaseLoader([
#     input_doc_url   # LangChain Introduction
#     ]
# )

# url_data = web_loader.load()
# db2 = Chroma.from_documents(url_data, embeddings_model)
# print(url_data)



encoder = OpenAIEncoder()

#input_query = 'R-CLIPS 오류 로그 분석해서 원인을 말해줘'
input_query = question

# input_log = ''
# input_code = ''
# input_config = ''
rl = RouteLayer(encoder=encoder, routes=routes)

route = rl(input_query)
print(route)

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

if route.name == "error":
    print('에러 분석')
    uploaded_file = st.file_uploader("로그 파일을 업로드하세요")
    st.write("---")
    
    db = uploadFileToDB(uploaded_file)

    memory1 = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True, output_key="output")
    # memory2 = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True, output_key="output")

    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=[],
        llm=llm,
        max_iterations=3,
        early_stopping_method="generate",
        memory=memory1,
    )

    system_message = '시니어 개발자의 역할로써 오류 방안에 대한 원인과 해결 방법을 제시'

    new_prompt = agent.agent.create_prompt(system_message=system_message, tools=[])
    agent.agent.llm_chain.prompt = new_prompt

    result = agent(input_query)
    print(result)
    
    
elif route.name == "guide":
    print('수정 가이드')
    uploaded_file = st.file_uploader("가이드에 필요한 코드 파일을 업로드하세요")
    st.write("---")

    db = uploadFileToDB(uploaded_file)

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm) 
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    result = qa_chain({"query":question})
    st.write(result["result"])
    
else:
    print('질문유형 벗어난 경우')
    pass



# llm = semantic_layer(input_query)
# route_reuslt = rl(input_query)


