from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI



class AgentState(TypedDict):
    user_input: str
    agent_output: str
    turn_count: int



llm = ChatGoogleGenerativeAI(
    api_key = "your-key",
    model="gemini-3-flash-review",
    temperature=0.3
)



def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["user_input"])

    return {
        "agent_output": response.content,
        "turn_count": state["turn_count"] + 1
    }



graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()


def main():
    print("LangGraph Gemini Agent (type 'exit' to quit)\n")

    turn_count = 0

    while True:
        user_text = input("You: ")

        if user_text.lower() == "exit":
            print("Exiting agent.")
            break

        state = {
            "user_input": user_text,
            "agent_output": "",
            "turn_count": turn_count
        }

        result = app.invoke(state)
        turn_count = result["turn_count"]

        print(f"Agent (turn {turn_count}): {result['agent_output']}\n")


if __name__ == "__main__":
    main()

