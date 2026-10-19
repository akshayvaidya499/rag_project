from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import uvicorn
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool
from langserve import add_routes
from bs4 import SoupStrainer, BeautifulSoup
import requests
from openai import OpenAI
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS


load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
# llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
# llm2 = Groq()
# print(llm.invoke("sunny leone").content)
# for i in llm2.models.list().data:
#     print(i.id)

def getCurrancyValue(cur1, cur2, multiplier):
      url = f"https://paytm.com/tools/currency-converter/amount-{multiplier}-from-{cur1.lower()}-to-{cur2.lower()}/"
      data = requests.get(url=url).text
      result = BeautifulSoup(data, "html.parser")
        
      values = result.find_all("div", {"class": "hJLwA"})
      value = values[0].find_all("span")[0].text
      
      print(value)
      return value

@tool
def getCurrancy(country1:str, country2:str, multiplier: str)-> str:
    """
    LLM reads the function, parameters and perform operation.
    It returns the currency 
    """
    currancy = getCurrancyValue(country1, country2, multiplier)
    return currancy

loader = WebBaseLoader(web_path="https://bullions.co.in")

# loader = WebBaseLoader(
#     web_path="https://www.mmtcpamp.com/gold-silver-rate-today",
#     bs_kwargs={
#         "parse_only": SoupStrainer(
#             class_="sc-4c2df15c-14"
#         )
#     }
# )
#a = requests.get("https://bullions.co.in").text
doc = loader.load()
# print(doc[0].page_content[:500])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)
db = FAISS.from_documents(splits, OpenAIEmbeddings())
#print(db.similarity_search("Gold prices"))
retriever = db.as_retriever()
marketTool = create_retriever_tool(retriever=retriever, name="market_rate", description="Provide the market rate information")

agent = create_agent(model=llm, tools=[getCurrancy, marketTool],
                     system_prompt="""
                      You are helpful AI Assistant. Provide the answer to the user query.
                      if user query is about currency conversion, then first find out multiplier and official currency short form of the country name 
                      specified in the query.
                      For example, 4 indian ruppes to us dollar, answer is (4, INR, USD) or
                       australian dollar in indian rupees, answer is (1, AUD, INR)
                      Use this information to call the tool getCurrancy(country1, country2, multiplier) to get the answer. 
                      
                      if user query is about market rate(Gold and silver prices), use marketTool to get the answer.
                      
                      Keep earlier conversation messages in context when
                      answering follow-up questions.
                     """ )

app=FastAPI(title="Langchain Server",
        version="1.0",
        description="Simple API of server")

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai"
# )

prompt = ChatPromptTemplate.from_template(
    """
    You are helpful AI Assistant. Provide the answer to the user query. Keep earlier response as a context.
    <context>
    {context}
    </context>
    
    Question: {topic}                                      
    """)
#result = agent.invoke({"messages": [{"role": "user", "content": "How much is 5 australian dollar in indian rupees?"}]})
#print(result["messages"][-1].content)
add_routes(
    app,
    agent,
    path="/story"
)

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="localhost",
        port=8000
    )