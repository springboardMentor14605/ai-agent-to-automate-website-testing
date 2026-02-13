import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Set API key from environment variable (DO NOT hardcode)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# 1️⃣ Define State
class AgentState(TypedDict):
    user_input: str
    agent_response: str
    message_count: int   # Additional field


# 2️⃣ Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)


# 3️⃣ Define Node
def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        HumanMessage(content=state["user_input"])
    ])

    text = response.content

    return {
        "user_input": state["user_input"],
        "agent_response": text,
        "message_count": state["message_count"] + 1
    }


# 4️⃣ Build Graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()


# 5️⃣ Dynamic Loop
def main():
    message_count = 0

    while True:
        user_input = input("Enter something (or type 'exit'): ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = app.invoke({
            "user_input": user_input,
            "agent_response": "",
            "message_count": message_count
        })

        message_count = result["message_count"]

        print("\nAgent:", result["agent_response"])
        print("Messages processed:", message_count)
        print("-" * 40)


if __name__ == "__main__":
    main()
