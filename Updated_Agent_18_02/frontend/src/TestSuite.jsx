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
        instruction: "Fill first name 'John', last name 'Doe', job title 'QA Engineer', and phone '5551234567'"
    },
    {
        id: 2,
        title: "Sauce Demo - Login & Browse",
        description: "Login and explore products",
        website: "https://www.saucedemo.com",
        username: "standard_user",
        password: "secret_sauce",
        instruction: "Login and verify you can see at least 5 products on the page"
    },
    {
        id: 3,
        title: "NopCommerce - Search Product",
        description: "Add item to cart on demo store",
        website: "https://demo.nopcommerce.com/",
        username: "",
        password: "",
        instruction: "Search for 'laptop', click on first result, and add 1 to cart"
    },
    {
        id: 4,
        title: "Dropdown Selection",
        description: "Select from dropdown menu",
        website: "https://the-internet.herokuapp.com/dropdown",
        username: "",
        password: "",
        instruction: "Click on the dropdown menu and select 'Option 1'"
    },
    {
        id: 5,
        title: "Checkbox Interaction",
        description: "Toggle checkboxes",
        website: "https://formy.herokuapp.com/checkbox",
        username: "",
        password: "",
        instruction: "Check the checkbox labeled 'Check this checkbox'"
    },
    {
        id: 6,
        title: "Radio Button Selection",
        description: "Select radio button option",
        website: "https://formy.herokuapp.com/radio-button",
        username: "",
        password: "",
        instruction: "Click on the radio button for 'Option 1'"
    }
];

function TestSuite() {
    const [tests, setTests] = useState(PRESET_TESTS.map(t => ({ ...t, status: null, loading: false })));
    const [runningAll, setRunningAll] = useState(false);
    const [results, setResults] = useState(null);

    const runTest = async (testId) => {
        const updatedTests = tests.map(t =>
            t.id === testId ? { ...t, loading: true, status: null } : t
        );
        setTests(updatedTests);

        const test = PRESET_TESTS.find(t => t.id === testId);
        
        try {
            const response = await fetch('http://localhost:8000/api/run-test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: test.website,
                    username: test.username,
                    password: test.password,
                    phone: ''
                }),
            });

            const data = await response.json();
            const status = data.status === 'PASS' ? 'pass' : 'fail';

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
        for (const test of tests) {
            await runTest(test.id);
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        setRunningAll(false);
        const passed = tests.filter(t => t.status === 'pass').length;
        setResults({
            total: tests.length,
            passed,
            failed: tests.length - passed,
            successRate: Math.round((passed / tests.length) * 100)
        });
    };

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

            {results && (
                <div className="test-results-summary">
                    <div className="summary-card">
                        <div className="summary-stat">
                            <span className="stat-label">Total Tests</span>
                            <span className="stat-value">{results.total}</span>
                        </div>
                        <div className="summary-stat pass">
                            <span className="stat-label">Passed</span>
                            <span className="stat-value">{results.passed}</span>
                        </div>
                        <div className="summary-stat fail">
                            <span className="stat-label">Failed</span>
                            <span className="stat-value">{results.failed}</span>
                        </div>
                        <div className="summary-stat">
                            <span className="stat-label">Success Rate</span>
                            <span className="stat-value">{results.successRate}%</span>
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
                            <div className="detail">
                                <span className="detail-label">User:</span>
                                <span className="detail-value">{test.username}</span>
                            </div>
                        </div>

                        <p className="test-instruction">
                            <strong>Task:</strong> {test.instruction}
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
