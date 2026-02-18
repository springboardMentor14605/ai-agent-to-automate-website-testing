# Project Structure & Architecture

## Overview
This project provides an automated website testing solution using Playwright, driven by an LLM-based agent. It features a React frontend for user input and a FastAPI backend for execution.

## Directory Structure
- `backend/`: Contains the Python backend logic.
  - `main.py`: The FastAPI application entry point. Exposes `/api/run-test`.
  - `playwright_agent.py`: The core agent logic using LangGraph and Playwright.
  - `playwright_executor.py`: Helper class to execute Playwright commands and capture screenshots.
  - `agent/`: (Planned) Future home for agent specific logic.
  - `screenshots/`: Directory where test execution screenshots are saved.
- `frontend/`: Contains the React frontend application.
  - `src/App.jsx`: Main application component with the form and result display.
  - `src/index.css`: Global styles including dark mode and animations.

## Usage
1. **Backend**:
   - Navigate to `backend/`.
   - Install dependencies: `pip install -r requirements.txt`.
   - Run the server: `python main.py` (starts on http://localhost:8000).
2. **Frontend**:
   - Navigate to `frontend/`.
   - Install dependencies: `npm install`.
   - Run dev server: `npm run dev`.
   - Open browser at http://localhost:5173.

## Features
- **Input**: User provides URL, Credentials (Email/Username, Password).
- **Execution**: The backend uses Playwright to perform the login test.
- **Output**: Returns Pass/Fail status, Logs, and a Screenshot of the final state.
