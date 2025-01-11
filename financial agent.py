from phi.agent import Agent 
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai


import os
openai.api_key=os.getenv("") 

#Web search agent 
web_search_agent=Agent(

    name = "Web Search Agent",
    role = "Search the web for the information",
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tools_calls = True,
    markdown=True,


)

##Financial Agent

finance_agent = Agent(

    name = "Finance Search Agent",
    role = "Search the web for finacial info" , 
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True
        ,company_news=True)
        ],
    instructions = ["Use table to display the data"],
    show_tool_calls=True,
    markdown=True,

)

##Now when we combine these 2 agents ,it becomes a multimodal agent 

multi_ai_agent=Agent(
    team = [web_search_agent,finance_agent], #Combined both the agents
    instructions=["Always include sources" , "Use table to display the data " ] ,
    show_tool_calls= True,
    markdown=True, 

)



## Now in order to initiate the multi agent

multi_ai_agent.print_response("Sumarise the analyst recommendation and share the latest news for NVIDIA",stream=True)
