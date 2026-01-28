from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class AgentState(TypedDict):
    user_input: str
    agent_response: str

# Use ChatHuggingFace 
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

# Initialize the base endpoint (do not specify the task here)
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=1000,
    temperature=0.7,
)
chat_model = ChatHuggingFace(llm=llm)


def agent_node(state: AgentState) -> AgentState:
    response = chat_model.invoke([
        HumanMessage(content=state["user_input"])
    ])
    if isinstance(response.content, list):
        text = "".join(
            block["text"] for block in response.content
            if block.get("type") == "text"
        )
    else:
        text = response.content
    return {
        "user_input": state["user_input"],
        "agent_response": text
    }

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

result = app.invoke({
    "user_input": "give an entire playwright script written in javascript only to test a website with this url: https://practice.automationtesting.in/",
    "agent_response": ""
})

print(result["agent_response"])
