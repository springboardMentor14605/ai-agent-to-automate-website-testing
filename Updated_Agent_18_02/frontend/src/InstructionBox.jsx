import { useState, useEffect, useRef } from 'react';
import './InstructionBox.css';

const EXAMPLE_COMMANDS = [
    { label: "Add 2 iPhones to cart", icon: "" },
    { label: "Remove AirPods from cart", icon: "" },
    { label: "Click the 'Buy Now' button", icon: "" },
    { label: "Fill search box with 'laptop'", icon: "" },
    { label: "Navigate to checkout", icon: "" },
    { label: "Clear my cart", icon: "" },
];

const INTENT_ICONS = {
    navigate: "",
    search: "",
    add_to_cart: "",
    remove_from_cart: "",
    update_quantity: "",
    clear_cart: "",
    fill_form: "",
    click_element: "",
    login: "",
    checkout: "",
    custom: "",
    error: "",
};

function InstructionBox() {
    const [instruction, setInstruction] = useState('');
    const [url, setUrl] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [feedbackVisible, setFeedbackVisible] = useState(false);
    const textareaRef = useRef(null);
    const suggestionsRef = useRef(null);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
        }
    }, [instruction]);

    // Close suggestions when clicking outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (suggestionsRef.current && !suggestionsRef.current.contains(e.target)) {
                setShowSuggestions(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Show feedback animation
    useEffect(() => {
        if (result) {
            setFeedbackVisible(true);
        }
    }, [result]);

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!instruction.trim()) {
            setError('Please enter an instruction.');
            return;
        }
        if (!url.trim()) {
            setError('Please enter a target URL.');
            return;
        }

        setLoading(true);
        setResult(null);
        setError(null);
        setFeedbackVisible(false);

        try {
            const response = await fetch('http://localhost:8000/api/run-instruction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instruction: instruction.trim(),
                    url: url.trim(),
                }),
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.statusText}`);
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

    const handleSuggestionClick = (label) => {
        setInstruction(label);
        setShowSuggestions(false);
        if (textareaRef.current) {
            textareaRef.current.focus();
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const intentIcon = result ? (INTENT_ICONS[result.intent] || "") : "";

    return (
        <div className="instruction-section">
            {/* Section Header */}
            <div className="instruction-header">
                <div className="instruction-header-icon"></div>
                <div>
                    <h2>Instruction Box</h2>
                    <p>Type natural language commands to automate website actions</p>
                </div>
            </div>

            {/* Input Form */}
            <form onSubmit={handleSubmit} className="instruction-form">
                {/* URL Input */}
                <div className="instruction-form-group">
                    <label htmlFor="instruction-url">
                        <span className="label-icon"></span>
                        Target URL
                    </label>
                    <input
                        id="instruction-url"
                        type="text"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder="https://www.example.com"
                        className="instruction-url-input"
                        required
                    />
                </div>

                {/* Instruction Textarea */}
                <div className="instruction-form-group" ref={suggestionsRef}>
                    <label htmlFor="instruction-input">
                        <span className="label-icon"></span>
                        Your Instruction
                    </label>
                    <div className="instruction-textarea-wrapper">
                        <textarea
                            id="instruction-input"
                            ref={textareaRef}
                            value={instruction}
                            onChange={(e) => setInstruction(e.target.value)}
                            onFocus={() => setShowSuggestions(true)}
                            onKeyDown={handleKeyDown}
                            placeholder="Type a command like 'Add 2 iPhones to cart' or 'Click the Buy Now button'..."
                            className="instruction-textarea"
                            rows={2}
                        />
                        <div className="textarea-hint">
                            Press <kbd>Enter</kbd> to submit · <kbd>Shift+Enter</kbd> for new line
                        </div>
                    </div>

                    {/* Suggestions Dropdown */}
                    {showSuggestions && !instruction.trim() && (
                        <div className="suggestions-dropdown">
                            <div className="suggestions-title">💡 Try an example:</div>
                            <div className="suggestions-grid">
                                {EXAMPLE_COMMANDS.map((cmd, idx) => (
                                    <button
                                        key={idx}
                                        type="button"
                                        className="suggestion-chip"
                                        onClick={() => handleSuggestionClick(cmd.label)}
                                    >
                                        <span className="chip-icon">{cmd.icon}</span>
                                        {cmd.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Submit Button */}
                <button
                    type="submit"
                    disabled={loading}
                    className="instruction-submit-btn"
                >
                    {loading ? (
                        <>
                            <span className="instruction-loader"></span>
                            <span>Processing...</span>
                        </>
                    ) : (
                        <>
                            <span className="btn-icon">▶</span>
                            Execute Instruction
                        </>
                    )}
                </button>
            </form>

            {/* Loading State */}
            {loading && (
                <div className="instruction-loading-state">
                    <div className="loading-steps">
                        <div className="loading-step active">
                            <span className="step-dot"></span>
                            <span>Scouting page elements...</span>
                        </div>
                        <div className="loading-step">
                            <span className="step-dot"></span>
                            <span>Parsing instruction with AI...</span>
                        </div>
                        <div className="loading-step">
                            <span className="step-dot"></span>
                            <span>Executing in browser...</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Error Display */}
            {error && (
                <div className="instruction-error">
                    <span className="error-icon"></span>
                    {error}
                </div>
            )}

            {/* Result Display */}
            {result && feedbackVisible && (
                <div className={`instruction-result ${result.status === 'PASS' ? 'pass' : 'fail'}`}>
                    {/* Status Header */}
                    <div className="result-status-header">
                        <span className="result-status-icon">
                        </span>
                        <div className="result-status-text">
                            <h3>
                                {result.status === 'PASS' ? 'Instruction Executed Successfully' : 'Instruction Failed'}
                            </h3>
                            {result.description && (
                                <p className="result-description">{result.description}</p>
                            )}
                        </div>
                    </div>

                    {/* Intent & Entities Card */}
                    {result.intent && result.intent !== 'error' && (
                        <div className="result-intent-card">
                            <div className="intent-badge">
                                <span className="intent-icon">{intentIcon}</span>
                                <span className="intent-label">
                                    {result.intent.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </span>
                            </div>
                            {result.entities && Object.keys(result.entities).some(k => result.entities[k]) && (
                                <div className="entities-list">
                                    {result.entities.product && (
                                        <span className="entity-tag">
                                            <span className="entity-key">Product:</span> {result.entities.product}
                                        </span>
                                    )}
                                    {result.entities.quantity && (
                                        <span className="entity-tag">
                                            <span className="entity-key">Qty:</span> {result.entities.quantity}
                                        </span>
                                    )}
                                    {result.entities.action_type && (
                                        <span className="entity-tag">
                                            <span className="entity-key">Action:</span> {result.entities.action_type}
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Failure Reason */}
                    {result.status === 'FAIL' && result.error && (
                        <div className="instruction-failure-reason">
                            <h4>Reason for Failure</h4>
                            <p>{result.error}</p>
                        </div>
                    )}

                    {/* Parsed Steps */}
                    {result.steps_parsed && result.steps_parsed.length > 0 && (
                        <div className="parsed-steps-card">
                            <h4>Planned Steps</h4>
                            <ol className="parsed-steps-list">
                                {result.steps_parsed.map((step, idx) => (
                                    <li key={idx}>{step}</li>
                                ))}
                            </ol>
                        </div>
                    )}

                    {/* Screenshot */}
                    {result.screenshot_url && (
                        <div className="instruction-screenshot">
                            <h4>Screenshot</h4>
                            <img src={result.screenshot_url} alt="Instruction result" />
                        </div>
                    )}

                    {/* Execution Details */}
                    {result.details && result.details.length > 0 && (
                        <div className="instruction-details">
                            <h4>Execution Log</h4>
                            <ul className="execution-log">
                                {result.details
                                    .filter(d => !d.startsWith('[INFO]'))
                                    .map((detail, idx) => (
                                        <li
                                            key={idx}
                                            className={detail.startsWith('[PASS]') ? 'log-pass' : 'log-fail'}
                                        >
                                            {detail}
                                        </li>
                                    ))}
                            </ul>
                        </div>
                    )}

                    {/* Generated Code */}
                    {result.generated_code && (
                        <details className="instruction-code-accordion">
                            <summary>Generated Playwright Code</summary>
                            <pre>{result.generated_code}</pre>
                        </details>
                    )}
                </div>
            )}
        </div>
    );
}

export default InstructionBox;
