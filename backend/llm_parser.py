import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# API Key Configuration
api_key = os.getenv("ABHAY_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    print("Warning: ABHAY_API_KEY not found in environment settings. Make sure .env exists and has the key.")

# Initializing the Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0
)

SYSTEM_PROMPT = """
You are an instruction parser for an automated web testing system.
Your task:
- Convert a natural language test case into structured test commands.
- Output ONLY valid JSON.
- Do not explain anything.
- Do not add extra keys.
Allowed actions:
- open
- fill
- click
- assert
Output format:

[
{
"action": "<action>",
"params": { ... }
}
]
"""

def llm_parse_instruction(instruction):
    """
    Instruction Parsing Function
    This function sends the system prompt and user instruction to Gemini and handles all possible response formats
    returned by the model.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=instruction)
    ]
    response = llm.invoke(messages)
    content = response.content
    
    # Handle list content (multimodal/safety fallback)
    if isinstance(content, list):
        # Extract text from blocks if possible
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            else:
                text_parts.append(str(block))
        content = "".join(text_parts)

    # Clean up code blocks if present (JSON markdown)
    if isinstance(content, str):
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        elif "```" in content:
            content = content.replace("```", "")
        content = content.strip()

    try:
        if isinstance(content, str):
            # Attempt to parse JSON
            return json.loads(content)
    except json.JSONDecodeError:
        pass
        
    # Fallback checks from the prompt
    if isinstance(content, dict) and "text" in content:
        return json.loads(content["text"])
    if isinstance(content, list):
        return content
        
    # If json.loads succeeded above, it returns. If not, we might be here.
    return content 

if __name__ == "__main__":
    test_case = """
    Open the login page.
    Enter username as admin.
    Enter password as 1234.
    Click the login button.
    """

    try:
        output = llm_parse_instruction(test_case)
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"Error: {e}")