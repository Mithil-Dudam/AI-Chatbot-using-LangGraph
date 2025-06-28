from langchain_ollama import OllamaLLM
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import START,END, StateGraph

llm = OllamaLLM(model='llama3.2')

class State(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state:State):
    return{"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

user_input = input("Enter Message: ")
state = graph.invoke({"messages":[{"role":"user","content":user_input}]})

print(state["messages"][-1].content)