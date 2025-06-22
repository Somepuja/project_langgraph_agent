from typing import Annotated 

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages 

import os 
from dotenv import load_dotenv 

from langchain_groq import ChatGroq

from IPython.display import Image, display

from langchain_tavily import TavilySearch

from langgraph.prebuilt import ToolNode 
from langgraph.prebuilt import tools_condition

# from langgraph.checkpoint.memory import MemorySaver

## loading env variables
load_dotenv()

## Define state of the agent
class State(TypedDict):
    """
    Messages have the type "list". The "add_messages" function
    in the annotation defines how this statekey should be updated
    (in this case, It appends messages to the list, rather than overwriting them)
    """
    messages:Annotated[list,add_messages]

## model 
llm = ChatGroq(model = "llama-3.1-8b-instant")

## define tool1 : TavilySearch
search_internet = TavilySearch(max_results=2)
## define tool2 : multiply two integer
def multiply(a: int, b: int) -> int:
    """Multiply a and b
    Args:
    a(int) : first int
    b(int) : second int

    Returns:
    int : product of a and b
    """
    return a * b

## create list of all tools
tools = [search_internet, multiply]

## bind llm with tools
llm_with_tools = llm.bind_tools(tools)

# initialize memory 
# memory = MemorySaver()

## model with tools
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

## build the graph 
builder = StateGraph(State)
## add nodes to graph
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))
## add edges to graph
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools","tool_calling_llm")

## compile the graph
graph = builder.compile() # graph = builder.compile(checkpointer=memory)

## Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))


## use memory
# config = {"configurable":{"thread_id":"1"}}

# response1 = graph.invoke({"messages":"Hi my name is Anjali"},config=config)
# response2 = graph.invoke({"messages":"What is my name?"},config=config)

## create response
response3 = graph.invoke({"messages":"tell me 2 latest news on AI and then multiply 6 with 7"})

# print(response1['messages'])
# print(response2['messages'])

## print response
print(response3['messages'][-1].content)

