from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# loader = PyPDFLoader("nutritionrater.pdf")
# doc = loader.load()
#retrieval_chain = Runnable()

# doc2 = list()
#print(doc)
def create_rag_chain(path):
    #global retrieval_chain
    
    loader = PyPDFLoader(path)
    doc = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(doc)
    #print(splits)

    db = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={"k":4})
    #print(db)

    llm = ChatOpenAI(model="gpt-5-mini")

    prompts = ChatPromptTemplate.from_template(
        """You are helpful AI assistant. Provide the answer based on the context from the document.
        If user query is not matching with the context, simply say I don’t have enough information from the
        document.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """
    )

    chain = create_stuff_documents_chain(llm=llm, prompt=prompts)

    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain
# result = retrieval_chain.invoke({"input":"How to keep heart healthy?"})
# print(result["answer"]