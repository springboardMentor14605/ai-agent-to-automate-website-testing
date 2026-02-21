# AutoTest AI — Natural Language Website Testing Agent

> AI-powered end-to-end web testing using LangGraph, Google Gemini, and Playwright.

---

## Overview

**AutoTest AI** is an agentic QA automation tool that lets you test any website using **plain English instructions**. No Selenium knowledge, no XPath hunting — just describe what to test, and the AI handles the rest.

Under the hood, it uses a **3-node LangGraph pipeline**:

```
User Instructions (NL)
        │
        ▼
┌─────────────────────────────────────┐
│  Node 1: convert_user_instruction   │  ← Google Gemini (1 LLM call)
│          _to_actions                │    Parses NL → steps + assertions
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Node 2: execute_test_case          │  ← Playwright (headless Chromium)
│                                     │    Runs all steps in real browser
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Node 3: generate_test_report       │  ← Pass/Fail report compiled
│                                     │    Returned to UI
└─────────────────────────────────────┘
```

---

## Project Structure

```
autotest_ai/
│
├── app.py                      # Flask server + LangGraph agent pipeline
├── llm_assertion_generator.py  # Gemini LLM wrapper for assertion generation
├── playwright_executor.py      # Playwright headless browser test runner
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── static/
    └── index.html              # Frontend UI (dark-themed single-page app)
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **AI-Powered Parsing** | Google Gemini converts plain English into structured test steps |
| 🎭 **Real Browser Testing** | Playwright runs tests in a real headless Chromium browser |
| 🔗 **LangGraph Pipeline** | 3-node agentic graph with error handling and conditional routing |
| ✅ **Auto Assertions** | Playwright `expect()` assertions generated automatically per step |
| 📊 **Live Report UI** | Animated pass/fail report with per-step details and error messages |
| ⚡ **Single LLM Call** | Entire parse + assertion generation happens in one Gemini call |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **AI / LLM:** Google Gemini (`gemini-3-flash-preview`) via LangChain
- **Agent Framework:** LangGraph (`StateGraph`)
- **Browser Automation:** Playwright (Python, headless Chromium)
- **Frontend:** Vanilla HTML/CSS/JS (dark UI, no framework)

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- A **Google Gemini API key** (get one free at [aistudio.google.com](https://aistudio.google.com))

### Step 1 — Clone the repo and navigate to this folder

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/autotest_ai
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Playwright browsers

```bash
playwright install chromium
```

### Step 5 — Add your Gemini API key

Open `app.py` and replace the placeholder:

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

### Step 6 — Run the app

```bash
python app.py
```

Then open your browser at: **[http://localhost:5000](http://localhost:5000)**

---

## 🚀 How to Use

1. Enter the **URL** of the website you want to test
2. Type your **test instructions** in plain English (one step per line)
3. Click **▶ Run Tests**
4. View the detailed pass/fail report

### Example Instructions

```
Enter username as student
Enter password as Password123
Click the Submit button
Verify that the page shows Congratulations
```

The AI will automatically:
- Parse these into `fill`, `fill`, `click`, `wait` actions
- Generate CSS selectors for each element
- Create Playwright assertions to verify outcomes
- Execute everything in a real browser

---

## 🧩 Code Walkthrough

### `app.py` — The LangGraph Agent

Defines three graph nodes and wires them together:

| Node | Role |
|---|---|
| `convert_user_instruction_to_actions` | Sends one prompt to Gemini, gets back JSON steps + assertions |
| `execute_test_case` | Calls `PlaywrightExecutor` to run all steps; marks each PASS/FAIL |
| `generate_test_report` | Aggregates results into the final report dict |

Conditional edges route to `handle_generation_error` if any node sets `state["error"]`.

### `llm_assertion_generator.py` — Gemini Wrapper

- Initializes `ChatGoogleGenerativeAI` with `gemini-3-flash-preview`
- `extract_text()` safely handles both string and list response formats
- Used directly inside `app.py` Node 1 for the single LLM call

### `playwright_executor.py` — Browser Runner

- Launches headless Chromium via `sync_playwright`
- Supports actions: `open`, `fill`, `click`, `wait`
- Evaluates assertion strings using Python's `eval()` in a sandboxed scope with `expect` and `page`
- On step failure, records error message and **continues** remaining steps

### `static/index.html` — Frontend

- Single-page dark UI built with vanilla JS
- Polls status messages during test execution
- Renders per-step accordion with assertions, values, and errors

---

## 📡 API Reference

### `POST /api/run-tests`

**Request body:**
```json
{
  "url": "https://example.com",
  "instructions": "Enter username as student\nClick the submit button"
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "instructions": "...",
  "total_steps": 4,
  "steps_executed": 4,
  "status": "PASS",
  "error": null,
  "summary": { "passed": 4, "failed": 0, "skipped": 0 },
  "steps": [
    {
      "step": 1,
      "action": "open",
      "target": "https://example.com",
      "value": null,
      "expected": "Page loads",
      "assertions": ["expect(page.locator(\"h1\")).to_be_visible()"],
      "status": "PASS",
      "error": null
    }
  ]
}
```

---

## 🔍 Supported Test Actions

| Action | What it does | Example target |
|---|---|---|
| `open` | Navigate to URL | `https://example.com` |
| `fill` | Type text into an input | `#username`, `input[name="email"]` |
| `click` | Click a button or link | `#submit`, `.btn-login` |
| `wait` | Pause for 1 second | `page` |

---

## ⚠️ Known Limitations

- Assertion strings are evaluated with `eval()` — inputs are LLM-generated (not user-supplied), but keep this in mind for production use
- Complex multi-tab or file-upload flows are not yet supported
- Gemini API rate limits apply on the free tier

---


## 📄 License

This project is for academic/demonstration purposes.
