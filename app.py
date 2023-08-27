import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
 
def main():
    load_dotenv()

    st.header("Chitchat classification chatbot")
 
 
    text= 'Just had a bad leak come from the AC - It’s not chitchat , Gotcha - It’s chitchat, Hi - It’s chitchat, Oh awesome - It’s chitchat, Can anyone lift out cup table please? - It’s not chitchat, It went 13 times back to back had to use emergency stop to get it to stop - It’s not chitchat'


    #creating chunks of input data
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)

    
    embeddings = OpenAIEmbeddings()

    # storing embeddings at vectorstore
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)


    # Accept user questions/query
    query = st.text_input("Enter the sentence :")
    question = f'Classify following text as Chitchat or not , only answer in yes or no \n {query}'

    if query:

        docs = VectorStore.similarity_search(query=question, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question= question)
            print(cb)
        st.write(response)
 
if __name__ == '__main__':
    main()
