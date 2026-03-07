# Automated Website Testing System

A robust automated testing platform powered by AI and real browser automation.

## Features
- <img src="assets/icons/ai.svg" alt="AI" width="18" height="18"> **AI-Powered**: Uses Google Gemini to understand natural language test descriptions
- <img src="assets/icons/browser.svg" alt="Browser" width="18" height="18"> **Real Browser Testing**: Playwright automation for authentic test execution
- <img src="assets/icons/reports.svg" alt="Reports" width="18" height="18"> **Detailed Reports**: Step-by-step logs with screenshots
- <img src="assets/icons/async.svg" alt="Async" width="18" height="18"> **Async Architecture**: High-performance async/await implementation
- <img src="assets/icons/design.svg" alt="Design" width="18" height="18"> **Minimal UI**: Clean white and black design for clarity

## <img src="assets/icons/rocket.svg" alt="Quick Start" width="18" height="18"> Quick Start

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
playwright install chromium

# Add your Google API key to .env
echo "YOUR_API_KEY=your_key_here" > .env

# Start server
python main.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Access Application
Open http://localhost:5173 in your browser.

## <img src="assets/icons/book.svg" alt="Usage" width="18" height="18"> Usage

1. **Enter Target URL**: The login page you want to test
2. **Provide Credentials**: Username/email and password
3. **Run Test**: Click "Run Test" and wait for results
4. **View Results**: See PASS/FAIL status with screenshot and detailed logs

## <img src="assets/icons/tools.svg" alt="Tech Stack" width="18" height="18"> Tech Stack

**Backend:**
- Python (FastAPI)
- Playwright (Async)
- LangGraph
- Google Gemini AI

**Frontend:**
- React
- Vite
- Modern CSS

## <img src="assets/icons/book.svg" alt="Requirements" width="18" height="18"> Requirements
- Python 3.8+
- Node.js 16+
- Google Gemini API Key

## <img src="assets/icons/book.svg" alt="Documentation" width="18" height="18"> Documentation
See [COMPLETE_GUIDE.md](./Documentation/COMPLETE_GUIDE.md) for detailed documentation.

## <img src="assets/icons/target.svg" alt="Example Tests" width="18" height="18"> Example Tests
- **SauceDemo**: https://www.saucedemo.com/ (standard_user / secret_sauce)
- **The Internet**: https://the-internet.herokuapp.com/login (tomsmith / SuperSecretPassword!)

## <img src="assets/icons/tools.svg" alt="Configuration" width="18" height="18"> Configuration

### Modify Test Selectors
Edit `backend/main.py` to customize the test template for different websites.

### Adjust Timeouts
Modify `playwright_executor.py` constructor:
```python
def __init__(self, headless=True, slow_mo=0, timeout=30000)
```

### Change UI Colors
Update CSS variables in `frontend/src/index.css`.

## <img src="assets/icons/bug.svg" alt="Troubleshooting" width="18" height="18"> Troubleshooting

**Tests failing?**
- Verify selectors match your target site
- Check network connectivity
- Increase timeout values

**Screenshots not showing?**
- Ensure backend runs on port 8000
- Check CORS settings
- Verify `screenshots/` directory exists

## <img src="assets/icons/book.svg" alt="Project Structure" width="18" height="18"> Project Structure
```
├── backend/              # Python/FastAPI backend
│   ├── main.py           # API server
│   ├── playwright_agent.py    # LangGraph orchestration
│   └── playwright_executor.py # Async test executor
├── frontend/             # React frontend
│   └── src/
│       ├── App.jsx       # Main component
│       └── *.css         # Styling
└── Documentation/        # Guides and docs
```

## 🤝 Contributing
Contributions are welcome! Please ensure:
- Code follows existing style
- Tests pass
- Documentation is updated

## 📜 License
Educational purposes.

---
**Version**: 2.0.0 (Async Architecture)  
**Last Updated**: February 2026
