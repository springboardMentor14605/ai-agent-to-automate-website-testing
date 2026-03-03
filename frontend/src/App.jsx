import { useState } from 'react';
import './App.css';
import InstructionBox from './InstructionBox';
// import TestSuite from './TestSuite';
// import TestingPanel from './TestingPanel';

function App() {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    phone: '',
    url: ''
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('login');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/run-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err.message === 'Failed to fetch') {
        setError('Cannot connect to the backend server. Make sure it is running on http://localhost:8000');
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const getFailureReason = (details) => {
    if (!details) return null;
    const infoLines = details.filter(d => d.startsWith('[INFO]'));
    return infoLines.map(l => l.replace('[INFO] ', '')).join('\n');
  };

  return (
    <div className="app-container">
      {/* ── Top Navigation ── */}
      <nav className="top-nav">
        <div className="nav-brand">
          <div className="nav-logo"></div>
          <div className="nav-brand-text">
            <span className="nav-brand-name">Web_Test_Agent</span>
            <span className="nav-brand-sub">by Abhay</span>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ── */}
      <section className="hero-section">
        <h1 className="hero-title">
          Automated Website Testing & Browser Interaction
        </h1>
        <div className="hero-meta">
          <span className="meta-pill">
            <span className="meta-pill-icon"></span> Instructions
          </span>
          <span className="meta-pill">
            <span className="meta-pill-icon"></span> Screenshots
          </span>
        </div>

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button
            className={`tab-btn ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => setActiveTab('login')}
          >
            <span className="tab-icon"></span>
            Login Test
          </button>
          <button
            className={`tab-btn ${activeTab === 'instruction' ? 'active' : ''}`}
            onClick={() => setActiveTab('instruction')}
          >
            <span className="tab-icon"></span>
            Instruction Box
          </button>
          {/* <button
            className={`tab-btn ${activeTab === 'test-suite' ? 'active' : ''}`}
            onClick={() => setActiveTab('test-suite')}
          >
            <span className="tab-icon"></span>
            Test Suite
          </button>
          <button
            className={`tab-btn ${activeTab === 'testing-panel' ? 'active' : ''}`}
            onClick={() => setActiveTab('testing-panel')}
          >
            <span className="tab-icon"></span>
            Testing Panel
          </button> */}
        </div>
      </section>

      {/* ── Main Content ── */}
      <main className="main-content">

        {/* Login Test Tab */}
        {activeTab === 'login' && (
          <>
            <form onSubmit={handleSubmit} className="input-form">
              <div className="form-group">
                <label>Target URL</label>
                <input
                  type="text"
                  name="url"
                  value={formData.url}
                  onChange={handleChange}
                  placeholder="https://www.xyz.com"
                  required
                />
              </div>

              <div className="form-group">
                <label>Username / Email</label>
                <input
                  type="text"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  placeholder="Enter your username"
                  required
                />
              </div>

              <div className="form-group">
                <label>Password</label>
                <input
                  type="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="••••••••"
                  required
                />
              </div>

              <div className="form-group">
                <label>Phone Number (Optional)</label>
                <input
                  type="tel"
                  name="phone"
                  value={formData.phone}
                  onChange={handleChange}
                  placeholder="+91 00000 - 00000"
                />
              </div>

              <button type="submit" disabled={loading} className="submit-btn">
                {loading ? (
                  <>
                    <span className="loader"></span>
                    <span style={{ marginLeft: '8px' }}>Running Test...</span>
                  </>
                ) : 'Run Test →'}
              </button>
            </form>

            {/* Info Cards */}
            <div className="info-cards">
              <div className="info-card">
                <div className="info-card-icon"></div>
                <div className="info-card-label">Target</div>
                <div className="info-card-value">Any Website URL</div>
              </div>
              <div className="info-card">
                <div className="info-card-icon"></div>
                <div className="info-card-label">Testing</div>
                <div className="info-card-value">Login Verification</div>
              </div>
              <div className="info-card">
                <div className="info-card-icon"></div>
                <div className="info-card-label">Output</div>
                <div className="info-card-value">Screenshots & Logs</div>
              </div>
              <div className="info-card">
                <div className="info-card-icon"></div>
                <div className="info-card-label">Engine</div>
                <div className="info-card-value">Playwright</div>
              </div>
            </div>

            {error && <div className="error-box">{error}</div>}

            {result && (
              <div className={`result-box ${result.status === 'PASS' ? 'pass' : 'fail'}`}>
                <div className="status-header">
                  <h2>
                    Status: {result.status}
                  </h2>
                </div>

                {result.status === 'FAIL' && result.error && (
                  <div className="failure-reason">
                    <h3>Reason for Failure</h3>
                    <p>{result.error}</p>
                    {getFailureReason(result.details) && (
                      <p className="failure-detail">{getFailureReason(result.details)}</p>
                    )}
                  </div>
                )}

                {result.screenshot_url && (
                  <div className="screenshot-container">
                    <h3>Screenshot</h3>
                    <img src={result.screenshot_url} alt="Test Result Screenshot" />
                  </div>
                )}

                {result.details && result.details.length > 0 && (
                  <div className="details-container">
                    <h3>Execution Steps</h3>
                    <ul className="details-list">
                      {result.details
                        .filter(d => !d.startsWith('[INFO]'))
                        .map((detail, idx) => (
                          <li key={idx} className={detail.startsWith('[PASS]') ? 'success' : 'error'}>
                            {detail}
                          </li>
                        ))}
                    </ul>
                  </div>
                )}

                <details className="code-accordion">
                  <summary>Generated Code</summary>
                  <pre>{result.generated_code || "No code generated."}</pre>
                </details>
              </div>
            )}
          </>
        )}

        {/* Instruction Box Tab */}
        {activeTab === 'instruction' && (
          <InstructionBox />
        )}

        {/* Test Suite Tab */}
        {activeTab === 'test-suite' && (
          <TestSuite />
        )}

        {/* Testing Panel Tab */}
        {activeTab === 'testing-panel' && (
          <TestingPanel />
        )}
      </main>
    </div>
  );
}

export default App;
