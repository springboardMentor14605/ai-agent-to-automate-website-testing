from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    user_input: str
    agent_response: str
    message_count: int  # extra field

def agent_node(state: AgentState):
    user_text = state["user_input"]
    count = state["message_count"]

    response = f"Response {count}: You said '{user_text}'"

    return {
        "user_input": user_text,
        "agent_response": response,
        "message_count": count + 1
    }

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

# Loop for dynamic input
message_counter = 1

while True:
    user_text = input("Enter something (or type 'exit'): ")

    if user_text.lower() == "exit":
        break

    result = app.invoke({
        "user_input": user_text,
        "agent_response": "",
        "message_count": message_counter
    })

    print("Agent:", result["agent_response"])
    message_counter += 1
