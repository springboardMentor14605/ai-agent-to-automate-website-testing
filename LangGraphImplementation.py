import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Set your API key securely (better to use environment variables outside the script)
os.environ["GOOGLE_API_KEY"] = ""

class AgentState(TypedDict):
    user_input: str
    agent_response: str 

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke([HumanMessage(content=state["user_input"])])
    
    if isinstance(response.content, list):
        text = "".join(
            block.get("text", "") for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = response.content
    
    return {
        "user_input": state["user_input"],
        "agent_response": text
    }

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
app = graph.compile()

print("Type 'exit' anytime to quit the program.\n")

while True:
    user = input("User : ")
    if user.lower() == "exit":
        print(" AI: Good Bye \n\n")
        print("System: You typed 'exit' — program has ended.")
        break
    
    result = app.invoke({"user_input": user})
    print(f" AI: {result['agent_response']} \n\n")