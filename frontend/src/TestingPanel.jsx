// import { useState } from 'react';
// import './TestingPanel.css';

// const TEST_SCENARIOS = [
//     {
//         id: 'scenario-1',
//         category: 'E-Commerce',
//         title: "Sauce Demo - Login & Add to Cart",
//         complexity: "Advanced",
//         type: "login",
//         instruction: "",
//         website: "https://www.saucedemo.com",
//         username: "standard_user",
//         password: "secret_sauce"
//     },
//     {
//         id: 'scenario-2',
//         category: 'Form Handling',
//         title: "Multi-Field Form Submission",
//         complexity: "Hard",
//         type: "instruction",
//         instruction: "Fill the first name field with 'Alice', the last name field with 'Smith', and the job title field with 'Engineer'",
//         website: "https://formy.herokuapp.com/form",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-3',
//         category: 'E-Commerce',
//         title: "NopCommerce - Product Search",
//         complexity: "Medium",
//         type: "instruction",
//         instruction: "Type 'phone' into the search box and then click the search button to find products",
//         website: "https://demo.nopcommerce.com/",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-4',
//         category: 'UI Elements',
//         title: "Handle Dropdown Selection",
//         complexity: "Medium",
//         type: "instruction",
//         instruction: "Select 'Option 2' from the dropdown element with id 'dropdown'",
//         website: "https://the-internet.herokuapp.com/dropdown",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-5',
//         category: 'Form Handling',
//         title: "Checkbox Toggle Test",
//         complexity: "Easy",
//         type: "instruction",
//         instruction: "Click the first checkbox on the checkboxes page to toggle it",
//         website: "https://the-internet.herokuapp.com/checkboxes",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-6',
//         category: 'UI Elements',
//         title: "Button Click Test",
//         complexity: "Easy",
//         type: "instruction",
//         instruction: "Click on the 'Elemental Selenium' link at the bottom of the page",
//         website: "https://the-internet.herokuapp.com/",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-7',
//         category: 'Navigation',
//         title: "Multi-Window Handling",
//         complexity: "Hard",
//         type: "instruction",
//         instruction: "Click the 'Click Here' link on the page to trigger a new window",
//         website: "https://the-internet.herokuapp.com/windows",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-8',
//         category: 'E-Commerce',
//         title: "NopCommerce - Category Browse",
//         complexity: "Medium",
//         type: "instruction",
//         instruction: "Type 'shirt' into the search input field and press the search button",
//         website: "https://demo.nopcommerce.com/",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-9',
//         category: 'Form Handling',
//         title: "Autocomplete Form Test",
//         complexity: "Medium",
//         type: "instruction",
//         instruction: "Type '123 Main Street' into the address field on the autocomplete form page",
//         website: "https://formy.herokuapp.com/autocomplete",
//         username: "",
//         password: ""
//     },
//     {
//         id: 'scenario-10',
//         category: 'E-Commerce',
//         title: "Sauce Demo - Negative Login",
//         complexity: "Easy",
//         type: "login",
//         instruction: "",
//         website: "https://www.saucedemo.com",
//         username: "locked_out_user",
//         password: "secret_sauce"
//     }
// ];

// const COMPLEXITY_COLORS = {
//     'Easy': '#10b981',
//     'Medium': '#f59e0b',
//     'Hard': '#ef4444',
//     'Advanced': '#8b5cf6'
// };

// function TestingPanel() {
//     const [selectedScenario, setSelectedScenario] = useState(null);
//     const [filterCategory, setFilterCategory] = useState('All');
//     const [results, setResults] = useState({});

//     const categories = ['All', ...new Set(TEST_SCENARIOS.map(s => s.category))];

//     const filteredScenarios = filterCategory === 'All'
//         ? TEST_SCENARIOS
//         : TEST_SCENARIOS.filter(s => s.category === filterCategory);

//     const runScenario = async (scenario) => {
//         setResults(prev => ({ ...prev, [scenario.id]: { loading: true } }));

//         try {
//             let response;

//             if (scenario.type === 'login') {
//                 // Login-based scenario — use login endpoint
//                 response = await fetch('http://localhost:8000/api/run-test', {
//                     method: 'POST',
//                     headers: { 'Content-Type': 'application/json' },
//                     body: JSON.stringify({
//                         url: scenario.website,
//                         username: scenario.username,
//                         password: scenario.password,
//                         phone: ''
//                     }),
//                 });
//             } else {
//                 // Instruction-based scenario — use instruction endpoint
//                 response = await fetch('http://localhost:8000/api/run-instruction', {
//                     method: 'POST',
//                     headers: { 'Content-Type': 'application/json' },
//                     body: JSON.stringify({
//                         instruction: scenario.instruction,
//                         url: scenario.website,
//                     }),
//                 });
//             }

//             const data = await response.json();

//             // For negative login tests (locked_out_user), FAIL is expected behavior
//             let status;
//             if (scenario.id === 'scenario-10') {
//                 status = data.status === 'FAIL' ? 'pass' : 'fail';
//             } else {
//                 status = data.status === 'PASS' ? 'pass' : 'fail';
//             }

//             setResults(prev => ({
//                 ...prev,
//                 [scenario.id]: {
//                     loading: false,
//                     status,
//                     data
//                 }
//             }));
//         } catch (err) {
//             setResults(prev => ({
//                 ...prev,
//                 [scenario.id]: {
//                     loading: false,
//                     status: 'fail',
//                     error: err.message
//                 }
//             }));
//         }
//     };

//     return (
//         <div className="testing-panel">
//             <div className="panel-header">
//                 <h2>Testing Scenarios Panel</h2>
//                 <p>10 diverse real-world test scenarios to validate and improve automation</p>
//             </div>

//             {/* Filter */}
//             <div className="filter-section">
//                 <label>Filter by Category:</label>
//                 <div className="filter-buttons">
//                     {categories.map(cat => (
//                         <button
//                             key={cat}
//                             className={`filter-btn ${filterCategory === cat ? 'active' : ''}`}
//                             onClick={() => setFilterCategory(cat)}
//                         >
//                             {cat}
//                         </button>
//                     ))}
//                 </div>
//             </div>

//             {/* Scenarios Grid */}
//             <div className="scenarios-container">
//                 <div className="scenarios-list">
//                     {filteredScenarios.map(scenario => (
//                         <div
//                             key={scenario.id}
//                             className={`scenario-item ${selectedScenario?.id === scenario.id ? 'selected' : ''}`}
//                             onClick={() => setSelectedScenario(scenario)}
//                         >
//                             <div className="scenario-item-header">
//                                 <h3>{scenario.title}</h3>
//                                 <span
//                                     className="complexity-badge"
//                                     style={{ backgroundColor: COMPLEXITY_COLORS[scenario.complexity] }}
//                                 >
//                                     {scenario.complexity}
//                                 </span>
//                             </div>
//                             <p className="scenario-category">{scenario.category}</p>

//                             <div className="scenario-result">
//                                 {results[scenario.id]?.loading && (
//                                     <span className="result-status loading">
//                                         <span className="loader-small"></span> Running...
//                                     </span>
//                                 )}
//                                 {results[scenario.id]?.status === 'pass' && (
//                                     <span className="result-status pass">✓ Passed</span>
//                                 )}
//                                 {results[scenario.id]?.status === 'fail' && (
//                                     <span className="result-status fail">✗ Failed</span>
//                                 )}
//                             </div>
//                         </div>
//                     ))}
//                 </div>

//                 {/* Details Panel */}
//                 {selectedScenario && (
//                     <div className="scenario-details">
//                         <div className="details-header">
//                             <h3>{selectedScenario.title}</h3>
//                             <span
//                                 className="complexity-badge-large"
//                                 style={{ backgroundColor: COMPLEXITY_COLORS[selectedScenario.complexity] }}
//                             >
//                                 {selectedScenario.complexity}
//                             </span>
//                         </div>

//                         <div className="detail-section">
//                             <h4>Category</h4>
//                             <p>{selectedScenario.category}</p>
//                         </div>

//                         <div className="detail-section">
//                             <h4>Test Instruction</h4>
//                             <p className="instruction-text">
//                                 {selectedScenario.type === 'login'
//                                     ? `Login to ${selectedScenario.website} with user "${selectedScenario.username}"`
//                                     : selectedScenario.instruction}
//                             </p>
//                         </div>

//                         <div className="detail-section">
//                             <h4>Test Details</h4>
//                             <div className="details-grid">
//                                 <div>
//                                     <span className="detail-label">Website:</span>
//                                     <span>{selectedScenario.website}</span>
//                                 </div>
//                                 {selectedScenario.type === 'login' && (
//                                     <div>
//                                         <span className="detail-label">Username:</span>
//                                         <span>{selectedScenario.username}</span>
//                                     </div>
//                                 )}
//                             </div>
//                         </div>

//                         <div className="detail-section">
//                             <h4>Learning Outcomes</h4>
//                             <ul className="learning-outcomes">
//                                 <li>Tests NLP instruction parsing accuracy</li>
//                                 <li>Validates state management and session handling</li>
//                                 <li>Identifies edge cases and error scenarios</li>
//                                 <li>Measures generated code quality and correctness</li>
//                             </ul>
//                         </div>

//                         {results[selectedScenario.id] && (
//                             <div className={`result-display ${results[selectedScenario.id].status}`}>
//                                 <h4>Execution Result</h4>
//                                 <p>
//                                     Status: <strong>{results[selectedScenario.id].status?.toUpperCase()}</strong>
//                                 </p>
//                                 {results[selectedScenario.id].data?.screenshot_url && (
//                                     <img
//                                         src={results[selectedScenario.id].data.screenshot_url}
//                                         alt="Result screenshot"
//                                         className="result-screenshot"
//                                     />
//                                 )}
//                             </div>
//                         )}

//                         <button
//                             className="run-scenario-btn"
//                             onClick={() => runScenario(selectedScenario)}
//                             disabled={results[selectedScenario.id]?.loading}
//                         >
//                             {results[selectedScenario.id]?.loading ? 'Running...' : 'Run This Scenario'}
//                         </button>
//                     </div>
//                 )}
//             </div>
//         </div>
//     );
// }

// export default TestingPanel;
