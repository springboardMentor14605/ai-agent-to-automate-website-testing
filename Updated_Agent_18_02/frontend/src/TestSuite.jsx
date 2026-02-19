import { useState } from 'react';
import './TestSuite.css';

const PRESET_TESTS = [
    {
        id: 1,
        title: "Form Filling - Text Inputs",
        description: "Fill text input form fields",
        website: "https://formy.herokuapp.com/form",
        username: "",
        password: "",
        type: "instruction",
        instruction: "Fill the first name field with 'John' and the last name field with 'Doe'. Fill job title with 'QA Engineer'."
    },
    {
        id: 2,
        title: "Sauce Demo - Login",
        description: "Login with valid credentials",
        website: "https://www.saucedemo.com",
        username: "standard_user",
        password: "secret_sauce",
        type: "login",
        instruction: ""
    },
    {
        id: 3,
        title: "NopCommerce - Search Product",
        description: "Search for a product on demo store",
        website: "https://demo.nopcommerce.com/",
        username: "",
        password: "",
        type: "instruction",
        instruction: "Type 'laptop' into the search box and click the search button"
    },
    {
        id: 4,
        title: "Dropdown Selection",
        description: "Select from dropdown menu",
        website: "https://the-internet.herokuapp.com/dropdown",
        username: "",
        password: "",
        type: "instruction",
        instruction: "Select 'Option 1' from the dropdown with id 'dropdown'"
    },
    {
        id: 5,
        title: "Checkbox Interaction",
        description: "Toggle checkboxes on page",
        website: "https://the-internet.herokuapp.com/checkboxes",
        username: "",
        password: "",
        type: "instruction",
        instruction: "Click the first checkbox on the page to check it"
    },
    {
        id: 6,
        title: "Sauce Demo - Invalid Login",
        description: "Login with wrong password",
        website: "https://www.saucedemo.com",
        username: "standard_user",
        password: "wrong_password",
        type: "login",
        instruction: ""
    }
];

function TestSuite() {
    const [tests, setTests] = useState(PRESET_TESTS.map(t => ({ ...t, status: null, loading: false })));
    const [runningAll, setRunningAll] = useState(false);
    const [results, setResults] = useState(null);

    const runTest = async (testId) => {
        setTests(prevTests =>
            prevTests.map(t =>
                t.id === testId ? { ...t, loading: true, status: null } : t
            )
        );

        const test = PRESET_TESTS.find(t => t.id === testId);

        try {
            let response;

            if (test.type === 'login') {
                // Login test — use the login test endpoint
                response = await fetch('http://localhost:8000/api/run-test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: test.website,
                        username: test.username,
                        password: test.password,
                        phone: ''
                    }),
                });
            } else {
                // Instruction-based test — use the instruction endpoint
                response = await fetch('http://localhost:8000/api/run-instruction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        instruction: test.instruction,
                        url: test.website,
                    }),
                });
            }

            const data = await response.json();

            // For login tests with deliberately wrong credentials, a FAIL is expected
            let status;
            if (test.id === 6) {
                // Negative login test — FAIL from the engine means the test itself passed
                status = data.status === 'FAIL' ? 'pass' : 'fail';
            } else {
                status = data.status === 'PASS' ? 'pass' : 'fail';
            }

            setTests(prevTests =>
                prevTests.map(t =>
                    t.id === testId ? { ...t, status, loading: false, result: data } : t
                )
            );
        } catch (err) {
            setTests(prevTests =>
                prevTests.map(t =>
                    t.id === testId ? { ...t, status: 'fail', loading: false, error: err.message } : t
                )
            );
        }
    };

    const runAllTests = async () => {
        setRunningAll(true);
        // Reset all tests first
        setTests(prev => prev.map(t => ({ ...t, status: null, loading: false })));

        for (const test of PRESET_TESTS) {
            await runTest(test.id);
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        setRunningAll(false);
    };

    // Compute results after all tests complete
    const totalRun = tests.filter(t => t.status !== null).length;
    const passed = tests.filter(t => t.status === 'pass').length;
    const failed = tests.filter(t => t.status === 'fail').length;
    const showSummary = totalRun === tests.length && !runningAll;

    return (
        <div className="test-suite">
            <div className="test-suite-header">
                <h2>Test Suite Runner</h2>
                <button
                    className="run-all-btn"
                    onClick={runAllTests}
                    disabled={runningAll}
                >
                    {runningAll ? 'Running...' : 'Run All Tests'}
                </button>
            </div>

            {showSummary && (
                <div className="test-results-summary">
                    <div className="summary-card">
                        <div className="summary-stat">
                            <span className="stat-label">Total Tests</span>
                            <span className="stat-value">{tests.length}</span>
                        </div>
                        <div className="summary-stat pass">
                            <span className="stat-label">Passed</span>
                            <span className="stat-value">{passed}</span>
                        </div>
                        <div className="summary-stat fail">
                            <span className="stat-label">Failed</span>
                            <span className="stat-value">{failed}</span>
                        </div>
                        <div className="summary-stat">
                            <span className="stat-label">Success Rate</span>
                            <span className="stat-value">{Math.round((passed / tests.length) * 100)}%</span>
                        </div>
                    </div>
                </div>
            )}

            <div className="tests-grid">
                {tests.map(test => (
                    <div key={test.id} className={`test-card ${test.status}`}>
                        <div className="test-header">
                            <h3>{test.title}</h3>
                            <span className="test-id">#{test.id}</span>
                        </div>

                        <p className="test-description">{test.description}</p>

                        <div className="test-details">
                            <div className="detail">
                                <span className="detail-label">Website:</span>
                                <span className="detail-value">{test.website.replace('https://', '')}</span>
                            </div>
                            {test.type === 'login' && (
                                <div className="detail">
                                    <span className="detail-label">User:</span>
                                    <span className="detail-value">{test.username || '—'}</span>
                                </div>
                            )}
                        </div>

                        <p className="test-instruction">
                            <strong>Task:</strong> {test.type === 'login'
                                ? `Login with user "${test.username}"`
                                : test.instruction}
                        </p>

                        <div className="test-status">
                            {test.loading && (
                                <div className="status-loading">
                                    <span className="loader"></span> Running...
                                </div>
                            )}
                            {test.status === 'pass' && (
                                <div className="status-pass">
                                    <span className="status-icon">✓</span> PASSED
                                </div>
                            )}
                            {test.status === 'fail' && (
                                <div className="status-fail">
                                    <span className="status-icon">✗</span> FAILED
                                </div>
                            )}
                            {!test.status && !test.loading && (
                                <div className="status-pending">Not Run</div>
                            )}
                        </div>

                        <button
                            className="test-run-btn"
                            onClick={() => runTest(test.id)}
                            disabled={test.loading || runningAll}
                        >
                            {test.loading ? 'Running...' : 'Run Test'}
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default TestSuite;
