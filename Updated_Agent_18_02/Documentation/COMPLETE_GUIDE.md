# Automated Website Testing - Complete Guide

## 📋 Overview
This is an automated website testing system built with:
- **Backend**: Python, FastAPI, Playwright (async), LangGraph, Google Gemini AI
- **Frontend**: React with minimal white/black design
- **Architecture**: LLM-powered test generation with real browser execution

## 🎯 Key Features
1. **Natural Language Input**: Describe tests in plain English
2. **AI-Powered Parsing**: Gemini AI converts descriptions to test commands
3. **Real Browser Testing**: Uses Playwright for authentic browser automation
4. **Detailed Reporting**: Step-by-step execution logs with screenshots
5. **Async Architecture**: Non-blocking execution for better performance

## 📁 Project Structure
```
INFS_SPRNBRD/
├── backend/                           # Python Backend
│   ├── main.py                        # FastAPI server
│   ├── playwright_agent.py            # LangGraph agent orchestration
│   ├── playwright_executor.py         # Enhanced async executor
│   ├── llm_assertion_generator.py     # LLM-based assertion generation
│   ├── assertion_generator.py         # Rule-based assertion generation
│   ├── requirements.txt               # Python dependencies
│   ├── .env                           # API keys (ABHAY_API_KEY)
│   └── screenshots/                   # Test execution screenshots
│
├── frontend/                          # React Frontend
│   ├── src/
│   │   ├── App.jsx                    # Main application component
│   │   ├── App.css                    # Component styles
│   │   ├── index.css                  # Global styles
│   │   └── main.jsx                   # React entry point
│   ├── package.json                   # Node dependencies
│   └── vite.config.js                 # Vite configuration
│
└── Documentation/                     # Project documentation
    └── project_structure.md           # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Google Gemini API key

### Backend Setup
```powershell
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Create .env file with your API key
# Add: ABHAY_API_KEY=your_google_api_key_here

# Start the server
python main.py
# OR
uvicorn main:app --reload
```
Server runs on: http://localhost:8000

### Frontend Setup
```powershell
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
Application available at: http://localhost:5173

## 🎨 Design Philosophy
- **Minimal UI**: Clean white background with black text
- **Clarity First**: Clear typography and spacing
- **Focused Interactions**: Subtle hover states and transitions
- **Accessibility**: High contrast, readable fonts

## 🔧 How It Works

### 1. User Input Flow
```
User enters credentials → Frontend sends POST request →
Backend receives data → Agent processes input →
Playwright executes test → Results returned → UI displays outcome
```

### 2. Backend Components

#### **main.py** (FastAPI Server)
- Handles HTTP requests from frontend
- Manages CORS for local development
- Serves screenshot files statically
- Integrates with LangGraph agent

#### **playwright_agent.py** (LangGraph Orchestration)
```
parse_node → enrich_node → generate_node → execute_node
     ↓            ↓              ↓              ↓
Extract     Add           Generate       Run in
commands    assertions    async code     browser
```

#### **playwright_executor.py** (Enhanced Executor)
Key improvements:
- **Async/Await**: Full async Playwright API support
- **Timeout Handling**: Configurable timeouts (default 30s)
- **Wait Strategies**: Network idle, DOM loaded, visibility checks
- **Error Recovery**: Captures screenshots even on failure
- **Detailed Logging**: Step-by-step execution tracking

### 3. Test Execution Flow
1. **Parse**: LLM converts natural language to JSON commands
2. **Enrich**: Adds assertions based on expectations
3. **Generate**: Creates async Playwright Python code
4. **Execute**: Runs test in real  browser with:
   - Element visibility waits
   - Network idle detection
   - Screenshot capture
   - Detailed step logging

## 📊 API Reference

### POST `/api/run-test`
**Request Body:**
```json
{
  "username": "string",
  "password": "string",
  "phone": "string (optional)",
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "status": "PASS" | "FAIL",
  "error": "string | null",
  "screenshot_url": "string | null",
  "generated_code": "string",
  "details": [
    "✓ Step 1: open",
    "✓ Step 2: fill",
    "✗ Step 3: click - Timeout"
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

**1. "Playwright Sync API inside asyncio loop"**
✅ Fixed! We now use `async_playwright` throughout.

**2. Tests fail even with correct credentials**
- Check if selectors match the target website
- Verify network connectivity
- Increase timeout in `playwright_executor.py`

**3. Screenshots not displaying**
- Ensure backend is running on port 8000
- Check `screenshots/` directory exists
- Verify CORS settings in `main.py`

**4. Frontend doesn't connect to backend**
- Confirm both servers are running
- Check browser console for CORS errors
- Verify API URL in `App.jsx` (localhost:8000)

**5. "NotImplementedError" on Windows**
✅ Fixed! The code now sets `WindowsProactorEventLoopPolicy` automatically.

**Windows-Specific Note**: Async Playwright requires `WindowsProactorEventLoopPolicy` on Windows. This is automatically configured in both `main.py` and `playwright_agent.py`:
```python
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

**Testing the fix**: Run `python backend/test_async_fix.py` to verify async Playwright works on your system.

## 💡 Customization

### Changing Test Selectors
Edit `main.py` line 53-58 to modify the user input template:
```python
user_input = f"""
Open the login page at {req.url}.
Enter username as {req.username} in #your-selector.
Enter password as {req.password} in #your-password-selector.
Click the login button #your-button-selector.
"""
```

### Adjusting Timeouts
In `backend/playwright_executor.py`:
```python
def __init__(self, headless: bool = True, slow_mo: int = 0, timeout: int = 30000):
    self.default_timeout = timeout  # Change this value (milliseconds)
```

### Modifying UI Colors
Edit `frontend/src/index.css`:
```css
:root {
  --primary: #000000;     /* Primary color */
  --pass: #28a745;        /* Success color */
  --fail: #dc3545;        /* Error color */
}
```

## 📝 Testing Examples

### Example 1: SauceDemo Login
```javascript
URL: https://www.saucedemo.com/
Username: standard_user
Password: secret_sauce
```

### Example 2: The Internet Herokuapp
```javascript
URL: https://the-internet.herokuapp.com/login
Username: tomsmith
Password: SuperSecretPassword!
```

## 🎓 Architecture Decisions

### Why Async Playwright?
- ✅ Compatible with FastAPI's async event loop
- ✅ Better performance under load
- ✅ Non-blocking I/O operations
- ✅ Modern Python best practices

### Why LangGraph?
- ✅ Clear state management
- ✅ Easy to debug multi-step workflows
- ✅ Scalable agent orchestration
- ✅ Built for LLM integration

### Why React + Vite?
- ✅ Fast development experience
- ✅ Hot module replacement
- ✅ Modern build tooling
- ✅ Minimal boilerplate

## 🔐 Security Notes
- Never commit `.env` file to version control
- API keys should be stored in environment variables
- Consider rate limiting for production deployments
- Validate all user inputs before testing

## 🚢 Deployment Considerations
1. **Backend**: Use production ASGI server (Uvicorn with Gunicorn)
2. **Frontend**: Build with `npm run build` and serve statically
3. **Environment**: Set `headless=True` for production
4. **Monitoring**: Add logging and error tracking
5. **Scaling**: Consider containerization (Docker)

## 📚 Further Reading
- [Playwright Async API](https://playwright.dev/python/docs/api/class-playwright)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [React Best Practices](https://react.dev/)

## 🤝 Contributing
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Keep commits atomic and descriptive

## 📄 License
This project is for educational purposes.

---
**Last Updated**: February 2026
**Version**: 2.0.0 (Async Architecture)
