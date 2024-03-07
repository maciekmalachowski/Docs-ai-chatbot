import streamlit as st
from streamlit_chat import message
from langchain_experimental.agents import create_csv_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
# from plotai import PlotAI


def get_pdf_text(pdfs):
    text = ''
    for pdf in pdfs:
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

def get_vectorstore(text_chunks):
    embeddings = GPT4AllEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
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
            message(msg.content)

def button_callback():
    st.session_state.button_clicked = True

def input_callback():
    st.session_state.button_clicked = False

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

    if "file_type" not in st.session_state:
        st.session_state.file_type = None

    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False

    st.header("Ask your docs 📜")
    user_files = [st.file_uploader("Upload your docs", type=(["csv","pdf"]), on_change=input_callback)]

    if len(user_files) != 0:
        if st.button("Process", use_container_width=True, on_click=button_callback):
            if user_files[0].name.endswith(".pdf"):
                st.session_state.file_type = "pdf"
                with st.spinner("Processing..."):

                    # get pdf text
                    raw_text = get_pdf_text(user_files)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, llm)
                print('DONE')
            
            else:
        # TO DO CSV
                st.session_state.file_type = "csv"
                # agent = create_csv_agent(llm, user_files, verbose=True, agent_executor_kwargs={"handle_parsing_errors": True})
                # response = agent.run(user_question)
                # st.write(response)

        if st.session_state.button_clicked:
            # user question input
            user_question = st.text_input("Ask a question about your document: ")

            if user_question:
                if st.session_state.file_type == "pdf":
                    hadle_userinput(user_question)
                elif st.session_state.file_type == "csv":
                    agent = create_csv_agent(llm, user_files, verbose=True, agent_executor_kwargs={"handle_parsing_errors": True})
                    response = agent.run(user_question)
                    st.write(response)




# TO DO PLOTAI
            # if user_file.name.endswith(".csv"):
                # plot = PlotAI(user_file)





if __name__ == "__main__":
    main()