import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader


from dotenv import load_dotenv
load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")

faiss_store_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.9,
    max_tokens=500,
    streaming=True,
    verbose=True
)

urls_validation = all(url.strip() == "" for url in urls)
if process_url_clicked:
    if urls_validation:
        st.sidebar.error("❌ Please provide at least one valid URL before processing.")
    else:
        # load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        # split loaded data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # embeddigs and save it to FAISS index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_huggingface = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")

        with open(faiss_store_path, "wb") as f:
            pickle.dump(vectorstore_huggingface, f)


query = main_placeholder.text_input("Question: ")
if query and not urls_validation:
    if os.path.exists(faiss_store_path):
        with open(faiss_store_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            sources = result["sources"]
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
