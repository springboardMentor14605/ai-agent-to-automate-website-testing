from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",  
    max_new_tokens=100,
)

model = ChatHuggingFace(llm=llm)



class Chats(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


def Chat_bot(state:Chats):
    messages = state["messages"]
    response = model.invoke(messages)

    return {"messages":[response]}



checkpointer = MemorySaver()

graph = StateGraph(Chats)

graph.add_node("Chatbot",Chat_bot)

graph.add_edge(START,"Chatbot")
graph.add_edge("Chatbot",END)

chatbot = graph.compile(checkpointer=checkpointer)


while True:
    user = input("User: ")
    if user == "0":
        break
    config = {"configurable":{"thread_id":1}}
    response = chatbot.invoke({"messages":[HumanMessage(content=user)]},config=config)
    print("AI:   ",response["messages"][-1].content)