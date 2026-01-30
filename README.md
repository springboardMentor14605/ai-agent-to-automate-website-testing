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

### 1. `playwright_agent.py` (AI Agent)
The main agent script. It takes a natural language string (e.g., "Open login page, enter username..."), parses it into structured commands using an LLM, and generates ready-to-run Playwright code.

### 2. `test_playwright.py` (Sandbox)
A comprehensive test suite that benchmarks testing against a local `login.html` page. It includes:
*   Positive and Negative test scenarios.
*   **Auto-Screenshot**: Automatically saves screenshots to `screenshots/` if a test fails.

### 3. `login.html`
A responsive, local HTML login page used as the target for the sandbox tests.

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

### Running the AI Agent
1.  Open `playwright_agent.py`.
2.  Modify the `test_case` string in the `if __name__ == "__main__":` block with your desired instructions.
3.  Run the agent:
    ```bash
    python playwright_agent.py
    ```
4.  The generated code will be printed to the console and saved to `generated_test_script.py`.

### Running Sandbox Tests
Run the Pytest suite:
```bash
pytest test_playwright.py
```
*   **View Results**: Check console output for pass/fail status.
*   **Screenshots**: Check `screenshots/` folder for failure captures.
