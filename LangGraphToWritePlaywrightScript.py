from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re

os.environ["GOOGLE_API_KEY"] = "" 

class AgentState(TypedDict): 
    user_input: str
    agent_response: str

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

def agent_node(state: AgentState) -> AgentState:
    try:
        messages = [
            SystemMessage(content="You are an expert automation testing engineer specializing in Playwright with JavaScript."),
            HumanMessage(content=state["user_input"])
        ]
        
        response = llm.invoke(messages)

        if isinstance(response.content, list):
            text = response.content[0].get('text', '')
        else:
            text = response.content
        
        return {
            "user_input": state["user_input"],
            "agent_response": text
        }
    except Exception as e:
        return {
            "user_input": state["user_input"],
            "agent_response": f"Error: {str(e)}"
        }


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "user_input": "Generate a complete Playwright test script in JavaScript to test the website https://practice.automationtesting.in/. Include tests for: navigation, form filling, and page assertions.",
        "agent_response": ""
    })
    
 
    response_text = result["agent_response"]
    

    code_blocks = re.findall(r'```javascript\n(.*?)\n```', response_text, re.DOTALL)
    
    if code_blocks:
      
        test_script = max(code_blocks, key=len)
        
        with open("practice-site.spec.js", "w", encoding="utf-8") as f:
            f.write(test_script)
        
        print(" Test script saved to: practice-site.spec.js")
    
    print("="*80)
    print("GEMINI RESPONSE:")
    print("="*80)
    print(response_text)
    print("="*80)