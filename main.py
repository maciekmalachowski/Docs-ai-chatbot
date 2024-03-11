import streamlit as st
import pandas as pd
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader


def get_pdf_text(pdf):
    text = ''
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore_pdf(text_chunks):
    embeddings = GPT4AllEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_file_data(user_file):
    # we use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(user_file.getvalue())
        tmp_file_path = tmp_file.name
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
    return data

def get_vectorstore_csv(data):
    embeddings = GPT4AllEmbeddings()
    vectorstore = FAISS.from_documents(data, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, llm):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def hadle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg.content, is_user=True)
        else:
            if "ERROR" in msg.content:
                st.warning('ERROR: Input correct prompt.', icon="⚠️")
            else:
                message(msg.content)
    # st.session_state.input_disable = False

def button_callback():
    st.session_state.button_clicked = True

def uplader_callback():
    st.session_state.button_clicked = False

# def text_input_disable():
#     st.session_state.input_disable = True

def main():
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model="./docs/model/mistral-7b-instruct-v0.1.Q4_0.gguf", callbacks=callbacks)
    st.set_page_config(
        page_title="Docs AI Chatbot",
        page_icon="./docs/icon/bot.png", 
        layout="centered",
        initial_sidebar_state="collapsed",
        )

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
    if "csv_file_type" not in st.session_state:
        st.session_state.csv_file_type = False

    # if "input_disable" not in st.session_state:
    #     st.session_state.input_disable = False

    st.header("Ask your docs 📜")
    user_file = st.file_uploader("Upload your docs", type=(["csv","pdf"]), on_change=uplader_callback)

    if user_file:
        if st.button("Process", use_container_width=True, on_click=button_callback):
            if user_file.name.endswith(".pdf"):
                st.session_state.csv_file_type = False
                with st.spinner("Processing..."):
                    # get pdf text
                    raw_text = get_pdf_text(user_file)
                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # create vector store
                    vectorstore = get_vectorstore_pdf(text_chunks)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, llm)

            else:
                st.session_state.csv_file_type = True
                with st.spinner("Processing..."):
                    # get file data 
                    data = get_file_data(user_file)
                    # create vector store
                    vectorstore = get_vectorstore_csv(data)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, llm)
                

        if st.session_state.button_clicked:
            if st.session_state.csv_file_type:
                # display dataframe
                df = pd.DataFrame(pd.read_csv(user_file))
                st.dataframe(df)
            # user question input
            # user_question = st.text_input("Ask a question about your document: ", disabled=st.session_state.input_disable, on_change=text_input_disable)
            user_question = st.text_input("Ask a question about your document: ")
            if user_question:
                hadle_userinput(user_question)
                st.balloons()

                

if __name__ == "__main__":
    main()