from Rag import rag
import streamlit as st
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()

# if "question" not in st.session_state:
#     st.session_state.question = ""
    
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY") 
os.environ["LANGSMITH_TRACING"]  = os.getenv("LANGSMITH_TRACING")
st.title("Chatboat-RAG AI Assistant")

input_text = st.text_input("Ask the questions about the document", key="question")
a = None
st.sidebar.title("Upload a PDF")

file = st.sidebar.file_uploader("Upload the PDF file", type=["pdf"])

if file is not None:
   print("hi") 
   #st.session_state.file=(file) 
   with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file :
       tmp_file.write(file.getvalue())
       pdf_path = tmp_file.name
       print(pdf_path)
       a= rag.create_rag_chain(pdf_path)
       

         
 


if input_text and file is not None :
    result = a.invoke({"input":input_text})
    for i, doc in enumerate(result["context"], start=1):
        st.sidebar.write(f"**Chunk {i}**")
        st.sidebar.write(doc.page_content)
    st.write(result["answer"])
    # st.session_state.question = "" 
    

