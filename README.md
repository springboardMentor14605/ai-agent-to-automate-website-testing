# Playwright Automation Agent & Sandbox

## Purpose
This project provides an intelligent automated testing solution that combines:
1.  **AI Testing Agent**: A LangGraph-based agent that converts natural language test instructions into executable Playwright Python scripts using Google's Gemini LLM.
2.  **Testing Sandbox**: A robust tailored test suite (`test_playwright.py`) for verifying login functionality with features like automatic screenshot capture on failure.

## Tech Stack
*   **Python**: Core programming language.
*   **Playwright**: Browser automation library.
*   **Pytest**: Testing framework.
*   **LangChain / LangGraph**: Framework for building the AI agent.
*   **Google Gemini**: LLM used for parsing natural language instructions.
*   **HTML/CSS/JS**: For the local test application (`login.html`).

## Project Components

### Core Agents & Parsers
*   **`playwright_agent.py`**: The main AI agent script. It orchestrates the flow from natural language -> LLM parsing -> Code Generation -> Execution.
*   **`llm_parser.py`**: A specialized module that interacts with Google's Gemini API. It takes raw text instructions and returns structured JSON commands (open, click, fill, assert).
*   **`rule_based_parser.py`**: A deterministic fallback parser that uses keyword matching (e.g., "click", "fill") to parse simple instructions without using an LLM.

### Testing Sandbox (Manual & Automated)
*   **`test_playwright.py`**: The primary Pytest suite. It benchmarks testing against `login.html` and includes robust failure handling (screenshots).
*   **`conftest.py`**: Pytest configuration file that defines fixtures and hooks (like the screenshot-on-failure hook).
*   **`login.html`**: A responsive, local HTML login page used as the target for the sandbox tests.

### Demos & Utilities
*   **`demo1.py`**: A simple, standalone Playwright script (no Pytest) to demonstrate the basics of browser automation.
*   **`demo2.py`**: An advanced demo showing how to integrate Playwright with Pytest fixtures.
*   **`check_models.py`**: A utility script to verify your Google Gemini API key and list available models.

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/springboardMentor14605/ai-agent-to-automate-website-testing.git
    cd ai-agent-to-automate-website-testing
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    playwright install
    ```

3.  **Environment Configuration**:
    *   Create a `.env` file in the root directory.
    *   Add your Google Gemini API key:
        ```
        ABHAY_API_KEY=your_api_key_here
        # OR
        GOOGLE_API_KEY=your_api_key_here
        ```

## Usage

### 1. Running the AI Agent
To generate tests from natural language:
1.  Open `playwright_agent.py`.
2.  Modify the `test_case` string in the `if __name__ == "__main__":` block.
3.  Run the agent:
    ```bash
    python playwright_agent.py
    ```
4.  The generated code will be saved to `generated_test_script.py`.

### 2. Standard Testing (Sandbox)
Run the comprehensive Pytest suite:
```bash
pytest test_playwright.py
```
*   **View Results**: Check console output for pass/fail status.
*   **Screenshots**: Check `screenshots/` folder for failure captures.

### 3. Utilities
*   **Check API Key**: `python check_models.py`
*   **Test Rule Parser**: `python rule_based_parser.py`
*   **Run Basic Demo**: `python demo1.py`
