from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser

llm = OllamaLLM(model="llama3.2")

class MessageClassifier(BaseModel):
    message_type:Literal["emotional","logical"]=Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

class State(TypedDict):
    messages:Annotated[list, add_messages]
    message_type:str|None

def classify_message(state:State):
    last_message = state["messages"][-1]
    parser = PydanticOutputParser(pydantic_object=MessageClassifier)
    messages =[
        {"role":"system",
         "content":f"""Classify the user message as either:
            'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            'logical': if it asks for facts, information, logical analysis, or practical solutions
            Examples:
            User: I'm feeling really down and don't know what to do.
            Output: {{ "message_type": "emotional" }}

            User: What is the capital of Germany?
            Output: {{ "message_type": "logical" }}
         """
         },
         {"role":"user","content":last_message.content}
    ]
    raw = llm.invoke(messages)
    parsed = parser.invoke(raw)

    return {"message_type": parsed.message_type}

def router(state:State):
    message_type = state.get("message_type","logical")
    if message_type=="emotional":
        return{"next":"therapist"}
    return{"next":"logical"}

def therapist_agent(state:State):
    last_message = state["messages"][-1]
    messages = [
        {"role":"system",
         "content":"""
         You are a compassionate therapist. Focus on the emotional aspect of the user's message.
         Show empathy, validate their feelings, and help them process their emotions.
         Ask thoughtful questions to help them explore their feelings more deeply.
         Avoid giving logical solutions unless explicitly asked.             
         """
         },
         {"role":"user",
          "content":last_message.content
          }
    ]
    reply = llm.invoke(messages)
    return {"messages":[{"role":"assistant","content":reply}]}

def logical_agent(state:State):
    last_message = state["messages"][-1]
    messages = [
        {"role":"system",
         "content":"""
         You are a purely logical assistant. Focus only on facts and information.
         Provide clear, concise answers based on logic and evidence.
         Do not address emotions or provide emotional support.
         Be direct and straightforward in your responses.            
         """
         },
         {"role":"user",
          "content":last_message.content
          }
    ]
    reply = llm.invoke(messages)
    return {"messages":[{"role":"assistant","content":reply}]} 

graph_builder = StateGraph(State)

graph_builder.add_node("classifier",classify_message)
graph_builder.add_node("router",router)
graph_builder.add_node("therapist",therapist_agent)
graph_builder.add_node("logical",logical_agent)

graph_builder.add_edge(START,"classifier")
graph_builder.add_edge("classifier","router")
graph_builder.add_conditional_edges(
    "router",
    lambda state:state.get("next"),
    {"therapist":"therapist","logical":"logical"}
)
graph_builder.add_edge("therapist",END)
graph_builder.add_edge("logical",END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages":[],"message_type":None}
    while True:
        user_input=input("Message: ")
        if user_input=="exit":
            print("Bye")
            break
        state["messages"] = state.get("messages",[])+[
            {"role":"user","content":user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()