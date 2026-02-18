import { useState } from 'react';
import './TestingPanel.css';

const TEST_SCENARIOS = [
    {
        id: 'scenario-1',
        category: 'E-Commerce',
        title: "Complete Purchase Flow",
        complexity: "Advanced",
        instruction: "Login with standard_user, add 3 different items to cart, apply any discount if available, and complete the purchase",
        website: "https://www.saucedemo.com",
        username: "standard_user",
        password: "secret_sauce"
    },
    {
        id: 'scenario-2',
        category: 'Form Handling',
        title: "Multi-Step Form Submission",
        complexity: "Hard",
        instruction: "Fill all fields on form page with First Name 'Alice', Last Name 'Smith', Email 'alice@test.com', Phone '4155552671', Date '12292021', Textarea 'This is a test'",
        website: "https://formy.herokuapp.com/form",
        username: "",
        password: ""
    },
    {
        id: 'scenario-3',
        category: 'E-Commerce',
        title: "NopCommerce - Complete Shopping",
        complexity: "Hard",
        instruction: "Navigate to electronics, find a desktop computer product, click it, verify specifications, and add to cart",
        website: "https://demo.nopcommerce.com/",
        username: "",
        password: ""
    },
    {
        id: 'scenario-4',
        category: 'UI Elements',
        title: "Handle Multiple Dropdowns",
        complexity: "Medium",
        instruction: "Navigate to dropdown page and select 'Option 1' from the main dropdown",
        website: "https://the-internet.herokuapp.com/dropdown",
        username: "",
        password: ""
    },
    {
        id: 'scenario-5',
        category: 'Form Handling',
        title: "Checkbox Toggle Test",
        complexity: "Medium",
        instruction: "Visit checkbox page and check the checkbox labeled 'Check this checkbox'",
        website: "https://formy.herokuapp.com/checkbox",
        username: "",
        password: ""
    },
    {
        id: 'scenario-6',
        category: 'UI Elements',
        title: "Radio Button Selection",
        complexity: "Easy",
        instruction: "Navigate to radio button page and select 'Option 2'",
        website: "https://formy.herokuapp.com/radio-button",
        username: "",
        password: ""
    },
    {
        id: 'scenario-7',
        category: 'Navigation',
        title: "Multi-Window Handling",
        complexity: "Hard",
        instruction: "Click the 'Click Here' link on windows page to open new window and verify it opens",
        website: "https://the-internet.herokuapp.com/windows",
        username: "",
        password: ""
    },
    {
        id: 'scenario-8',
        category: 'E-Commerce',
        title: "Search & Add to Cart",
        complexity: "Medium",
        instruction: "Search for 'shirt' on nopcommerce, filter results, and add the first shirt to cart with quantity 2",
        website: "https://demo.nopcommerce.com/",
        username: "",
        password: ""
    },
    {
        id: 'scenario-9',
        category: 'Form Handling',
        title: "Date and Time Input",
        complexity: "Medium",
        instruction: "On formy form page, set date to 12/25/2023 and fill all required fields",
        website: "https://formy.herokuapp.com/form",
        username: "",
        password: ""
    },
    {
        id: 'scenario-10',
        category: 'E-Commerce',
        title: "Product Filter & Compare",
        complexity: "Advanced",
        instruction: "On nopcommerce, browse apparel category, filter by color, select multiple items, and view them in cart",
        website: "https://demo.nopcommerce.com/",
        username: "",
        password: ""
    }
];

const COMPLEXITY_COLORS = {
    'Easy': '#10b981',
    'Medium': '#f59e0b',
    'Hard': '#ef4444',
    'Advanced': '#8b5cf6'
};

function TestingPanel() {
    const [selectedScenario, setSelectedScenario] = useState(null);
    const [filterCategory, setFilterCategory] = useState('All');
    const [results, setResults] = useState({});

    const categories = ['All', ...new Set(TEST_SCENARIOS.map(s => s.category))];
    
    const filteredScenarios = filterCategory === 'All' 
        ? TEST_SCENARIOS 
        : TEST_SCENARIOS.filter(s => s.category === filterCategory);

    const runScenario = async (scenario) => {
        setResults(prev => ({ ...prev, [scenario.id]: { loading: true } }));

        try {
            const response = await fetch('http://localhost:8000/api/run-test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: scenario.website,
                    username: scenario.username,
                    password: scenario.password,
                    phone: ''
                }),
            });

            const data = await response.json();
            setResults(prev => ({
                ...prev,
                [scenario.id]: {
                    loading: false,
                    status: data.status === 'PASS' ? 'pass' : 'fail',
                    data
                }
            }));
        } catch (err) {
            setResults(prev => ({
                ...prev,
                [scenario.id]: {
                    loading: false,
                    status: 'fail',
                    error: err.message
                }
            }));
        }
    };

    return (
        <div className="testing-panel">
            <div className="panel-header">
                <h2>Testing Scenarios Panel</h2>
                <p>10 diverse real-world test scenarios to validate and improve automation</p>
            </div>

            {/* Filter */}
            <div className="filter-section">
                <label>Filter by Category:</label>
                <div className="filter-buttons">
                    {categories.map(cat => (
                        <button
                            key={cat}
                            className={`filter-btn ${filterCategory === cat ? 'active' : ''}`}
                            onClick={() => setFilterCategory(cat)}
                        >
                            {cat}
                        </button>
                    ))}
                </div>
            </div>

            {/* Scenarios Grid */}
            <div className="scenarios-container">
                <div className="scenarios-list">
                    {filteredScenarios.map(scenario => (
                        <div
                            key={scenario.id}
                            className={`scenario-item ${selectedScenario?.id === scenario.id ? 'selected' : ''}`}
                            onClick={() => setSelectedScenario(scenario)}
                        >
                            <div className="scenario-item-header">
                                <h3>{scenario.title}</h3>
                                <span
                                    className="complexity-badge"
                                    style={{ backgroundColor: COMPLEXITY_COLORS[scenario.complexity] }}
                                >
                                    {scenario.complexity}
                                </span>
                            </div>
                            <p className="scenario-category">{scenario.category}</p>
                            
                            <div className="scenario-result">
                                {results[scenario.id]?.loading && (
                                    <span className="result-status loading">
                                        <span className="loader-small"></span> Running...
                                    </span>
                                )}
                                {results[scenario.id]?.status === 'pass' && (
                                    <span className="result-status pass">✓ Passed</span>
                                )}
                                {results[scenario.id]?.status === 'fail' && (
                                    <span className="result-status fail">✗ Failed</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Details Panel */}
                {selectedScenario && (
                    <div className="scenario-details">
                        <div className="details-header">
                            <h3>{selectedScenario.title}</h3>
                            <span
                                className="complexity-badge-large"
                                style={{ backgroundColor: COMPLEXITY_COLORS[selectedScenario.complexity] }}
                            >
                                {selectedScenario.complexity}
                            </span>
                        </div>

                        <div className="detail-section">
                            <h4>Category</h4>
                            <p>{selectedScenario.category}</p>
                        </div>

                        <div className="detail-section">
                            <h4>Test Instruction</h4>
                            <p className="instruction-text">{selectedScenario.instruction}</p>
                        </div>

                        <div className="detail-section">
                            <h4>Test Details</h4>
                            <div className="details-grid">
                                <div>
                                    <span className="detail-label">Website:</span>
                                    <span>{selectedScenario.website}</span>
                                </div>
                                <div>
                                    <span className="detail-label">Username:</span>
                                    <span>{selectedScenario.username}</span>
                                </div>
                            </div>
                        </div>

                        <div className="detail-section">
                            <h4>Learning Outcomes</h4>
                            <ul className="learning-outcomes">
                                <li>Tests NLP instruction parsing accuracy</li>
                                <li>Validates state management and session handling</li>
                                <li>Identifies edge cases and error scenarios</li>
                                <li>Measures generated code quality and correctness</li>
                            </ul>
                        </div>

                        {results[selectedScenario.id] && (
                            <div className={`result-display ${results[selectedScenario.id].status}`}>
                                <h4>Execution Result</h4>
                                <p>
                                    Status: <strong>{results[selectedScenario.id].status?.toUpperCase()}</strong>
                                </p>
                                {results[selectedScenario.id].data?.screenshot_url && (
                                    <img
                                        src={results[selectedScenario.id].data.screenshot_url}
                                        alt="Result screenshot"
                                        className="result-screenshot"
                                    />
                                )}
                            </div>
                        )}

                        <button
                            className="run-scenario-btn"
                            onClick={() => runScenario(selectedScenario)}
                            disabled={results[selectedScenario.id]?.loading}
                        >
                            {results[selectedScenario.id]?.loading ? 'Running...' : 'Run This Scenario'}
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

export default TestingPanel;
