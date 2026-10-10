from fastapi import FastAPI
from langchain_openai import ChatOpenAI
import uvicorn
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool
from langserve import add_routes

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini")

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

agent = create_agent(model=llm, tools=[],
                     system_prompt="""
                      You are helpful AI Assistant. Provide the answer to the user query. Use tools if required.
                      if user query is about currency conversion, then first find out multiplier and official currency short form of the country name 
                        specified in the query.
                         in one line. 
                        For example, 4 indian ruppes to us dollar
                        answer is 4, INR, USD
                      Keep earlier response as a context.
                         <context>
                         {context}
                         </context>
                         
                         Question: {topic}
                     """ )

app=FastAPI(title="Langchain Server",
        version="1.0",
        description="Simple API of server")

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are helpful AI Assistant. Provide the answer to the user query. Keep earlier response as a context.
    <context>
    {context}
    </context>
    
    Question: {topic}                                      
    """)

add_routes(
    app,
    prompt|llm,
    path="/story"
)

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )