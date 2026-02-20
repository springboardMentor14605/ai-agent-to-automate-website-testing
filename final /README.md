# AI Test Agent - Website Testing Automation

An intelligent automation tool that leverages AI to automatically test websites and generate comprehensive reports. This project combines a React frontend with a Node.js backend powered by OpenAI and Playwright for web testing.

## Features

- **AI-Powered Test Generation**: Automatically generate test cases using OpenAI API
- **Web Automation**: Run browser tests using Playwright
- **Real-time Reporting**: View test results and reports in real-time
- **User-Friendly Interface**: React-based frontend for easy interaction
- **RESTful API**: Backend API for test execution and management

## Tech Stack

### Frontend
- **React** - UI library
- **Vite** - Build tool and dev server
- **CSS** - Styling
- **Axios/Fetch** - HTTP client for API calls

### Backend
- **Node.js** - JavaScript runtime
- **Express.js** - Web framework
- **Playwright** - Browser automation
- **OpenAI API** - AI-powered test generation

## Prerequisites

Before you begin, ensure you have the following installed:
- Node.js (v14 or higher)
- npm or yarn package manager
- OpenAI API key
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/springboardMentor14605/ai-agent-to-automate-website-testing.git
cd final
```

### 2. Setup Backend

```bash
cd backend
npm install
```

Create a `.env` file in the backend directory:
```
OPENAI_API_KEY=your_api_key_here
PORT=5000
```

### 3. Setup Frontend

```bash
cd ../frontend
npm install
```

## Usage

### Start the Backend Server

```bash
cd backend
npm start
```

The backend server will run on `http://localhost:5000`

### Start the Frontend Development Server

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in your terminal)

## Project Structure

```
final/
├── backend/
│   ├── src/
│   │   ├── index.js              # Main server entry point
│   │   ├── openaiClient.js       # OpenAI API integration
│   │   ├── playwrightService.js  # Playwright test automation
│   │   └── testController.js     # Test execution controller
│   └── package.json
├── frontend/
│   ├── src/
│   │   ├── main.jsx              # React entry point
│   │   ├── app.jsx               # Main app component
│   │   ├── app.css               # App styling
│   │   ├── api.js                # API client utilities
│   │   └── index.html            # HTML template
│   └── package.json
└── README.md                      # This file
```

## API Endpoints

### Test Execution
- **POST** `/api/test` - Start a new test
- **GET** `/api/test/:id` - Get test results by ID
- **GET** `/api/tests` - List all test runs

## Environment Variables

### Backend (.env)
```
OPENAI_API_KEY=your_openai_api_key
PORT=5000
NODE_ENV=development
```

## Running Tests

Tests can be triggered through the web interface. The system will:
1. Accept test parameters
2. Use OpenAI to generate test scenarios
3. Execute tests using Playwright
4. Return detailed results and reports

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify the PORT in your `.env` file.

### API Key Issues
Ensure your OpenAI API key is valid and has appropriate permissions.

### Dependencies Installation Failed
Try clearing npm cache:
```bash
npm cache clean --force
npm install
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Branch**: feature-manish  
**Last Updated**: February 2026
