from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
import os

load_dotenv()

def load_secret(name):
    try:
        if name in st.secrets:
            os.environ[name] = st.secrets[name]   
    except Exception:
        pass  

for key in ["GOOGLE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]:
    load_secret(key)

@st.cache_resource
def load_models():
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return model, embedding

model, embedding = load_models()


ask_template = PromptTemplate(
    template="""
    You are a study assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in the document."

Context:
{context}

Question: {question}

max_length = 1000 words!!
    """,
    input_variables=["context","question"]
)

summarize_template = PromptTemplate(
    template="""
You are a study assistant. Summarize the topic "{topic}" using ONLY the context below.
- Include important definitions, formulas, and key points if they appear in the context.
- Use a simple analogy where it helps explain a hard idea.
- Do not add information that is not in the context.
- If the context does not cover the topic, say "Not found in the document."

Context:
{context}

""",
input_variables=['topic','context']
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      
    chunk_overlap=150,
)


st.header("PDF Study Buddy")
st.write("Upload a PDF, then ask questions or request a summary. Answers are based only on the content of your document.")
file = st.file_uploader("Upload a PDF to get started", type="pdf")

@st.cache_resource(show_spinner="Reading and embedding your PDF...")
def get_embeddings(file_bytes):
    pages = PyPDFParser().parse(Blob.from_data(file_bytes))     
    docs = splitter.split_documents(pages)           
    document = [doc.page_content for doc in docs]
    document_embedding = embedding.embed_documents(document)
    return document_embedding, docs



def retrieve(query, k=3):
    query_embedding = embedding.embed_query(query)
    scores = cosine_similarity([query_embedding], document_embedding)[0]
    top_idx = np.argsort(scores)[::-1][:k]
    return [docs[i] for i in top_idx]



if file:
    document_embedding, docs = get_embeddings(file.getvalue())
    


    mode = st.selectbox("What would you like to do?",["Ask a question","Summarize a topic"])

    if mode == "Ask a question":
        ques = st.text_input("Your question", placeholder="e.g. What is supervised learning?")
        if st.button("Get answer"):
            try:
                top_docs = retrieve(ques)
                context = "\n\n".join(d.page_content for d in top_docs)
                prompt = ask_template.invoke({
                    "context":context,
                    "question":ques
                })
                result = model.invoke(prompt)
                st.write(result.content[0]['text'])
            except Exception as e:
                print(e)
                st.error("🚨 An error occured while calling the api!")
                
                

    elif mode == "Summarize a topic":
        qry = st.text_input("Topic to summarize", placeholder="e.g. Gradient descent")
        if st.button("Summarize"):
            try:
                top_docs = retrieve(qry)
                context = "\n\n".join(d.page_content for d in top_docs)
                prompt = summarize_template.invoke({
                        "topic":qry,
                        "context":context
                    })
                result = model.invoke(prompt)
                st.write(result.content[0]['text'])
            except Exception as e:
                print(e)
                st.error("🚨 An error occured while calling the api!")
                


else:
    st.info("Upload a PDF above to start studying.")
