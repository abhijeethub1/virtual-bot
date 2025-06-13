import { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TDSVirtualTA = () => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [healthStatus, setHealthStatus] = useState(null);

  // Check system health on component mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API}/health`);
      setHealthStatus(response.data);
    } catch (e) {
      console.error("Health check failed:", e);
      setHealthStatus({ status: "unhealthy", error: "Cannot connect to backend" });
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) { // 5MB limit
        setError("Image file must be smaller than 5MB");
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64String = e.target.result.split(',')[1]; // Remove data:image/...;base64, prefix
        setImage(base64String);
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearImage = () => {
    setImage(null);
    setImagePreview("");
    document.getElementById('imageInput').value = '';
  };

  const askQuestion = async () => {
    if (!question.trim()) {
      setError("Please enter a question");
      return;
    }

    setLoading(true);
    setError("");
    setAnswer(null);

    try {
      const payload = {
        question: question.trim(),
        ...(image && { image })
      };

      const response = await axios.post(`${API}/ask`, payload);
      setAnswer(response.data);
    } catch (e) {
      console.error("Error asking question:", e);
      if (e.response?.status === 503) {
        setError("Virtual TA service is not available. The index may not be built yet.");
      } else {
        setError(e.response?.data?.detail || "Failed to get an answer. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const sampleQuestions = [
    "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?",
    "How do I handle missing data in my assignment?",
    "What are the best practices for data visualization?",
    "How do I choose between different machine learning algorithms?",
    "What's the difference between MCAR, MAR, and MNAR?"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🤖 TDS Virtual Teaching Assistant
          </h1>
          <p className="text-gray-600 text-lg">
            AI-powered assistant for Tools in Data Science course at IIT Madras
          </p>
          
          {/* Health Status */}
          <div className="mt-4 inline-flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              healthStatus?.status === 'healthy' ? 'bg-green-500' :
              healthStatus?.status === 'partial' ? 'bg-yellow-500' : 'bg-red-500'
            }`}></div>
            <span className="text-sm text-gray-600">
              {healthStatus?.status === 'healthy' ? 'System Healthy' :
               healthStatus?.status === 'partial' ? 'Partial Service' : 'System Issues'}
            </span>
            {healthStatus?.documents_count && (
              <span className="text-xs text-gray-500">
                ({healthStatus.documents_count} documents indexed)
              </span>
            )}
          </div>
        </div>

        {/* Main Interface */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          {/* Question Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Ask your question:
            </label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your question about the TDS course..."
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows="3"
            />
          </div>

          {/* Image Upload */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Optional: Upload an image (screenshot, diagram, etc.)
            </label>
            <div className="flex items-center space-x-4">
              <input
                id="imageInput"
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {imagePreview && (
                <button
                  onClick={clearImage}
                  className="text-red-600 text-sm hover:text-red-800"
                >
                  Clear
                </button>
              )}
            </div>
            
            {/* Image Preview */}
            {imagePreview && (
              <div className="mt-4">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="max-w-xs max-h-48 rounded-lg border"
                />
              </div>
            )}
          </div>

          {/* Submit Button */}
          <button
            onClick={askQuestion}
            disabled={loading || !question.trim()}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-6 rounded-lg transition duration-200"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Processing...
              </div>
            ) : (
              'Ask Question'
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700">{error}</p>
            </div>
          )}
        </div>

        {/* Answer Display */}
        {answer && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">📝 Answer:</h3>
            <div className="prose max-w-none">
              <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
                {answer.answer}
              </p>
            </div>

            {/* Links */}
            {answer.links && answer.links.length > 0 && (
              <div className="mt-6">
                <h4 className="text-md font-semibold text-gray-800 mb-3">🔗 Related Discussions:</h4>
                <div className="space-y-2">
                  {answer.links.map((link, index) => (
                    <a
                      key={index}
                      href={link.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition duration-200"
                    >
                      <div className="text-blue-600 hover:text-blue-800 font-medium">
                        {link.text}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {link.url}
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Sample Questions */}
        {!answer && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">💡 Sample Questions:</h3>
            <div className="grid gap-3">
              {sampleQuestions.map((sampleQ, index) => (
                <button
                  key={index}
                  onClick={() => setQuestion(sampleQ)}
                  className="text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition duration-200"
                >
                  <span className="text-gray-700">{sampleQ}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>
            This virtual TA uses course content and discourse discussions to provide helpful answers.
            <br />
            For official course support, please contact the teaching staff.
          </p>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <TDSVirtualTA />
    </div>
  );
}

export default App;