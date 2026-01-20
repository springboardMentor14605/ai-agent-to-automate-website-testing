import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


os.environ["GOOGLE_API_KEY"] = "AIzaSyD1oOBhg3WUvjTzFN99L3GeaAxi_iWRfEQ"

class AgentState(TypedDict):
    user_input: str
    agent_response: str
    response_length: int   

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke([
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
        "agent_response": text,
        "response_length": len(text)  
    }


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

while True:
    user_text = input("\nEnter your question (type 'exit' to quit): ")

    if user_text.lower() == "exit":
        print("Exiting agent...")
        break

    result = app.invoke({
        "user_input": user_text,
        "agent_response": "",
        "response_length": 0
    })

    print("\nAgent Response:")
    print(result["agent_response"])
    print(f"\nResponse Length: {result['response_length']} characters")
