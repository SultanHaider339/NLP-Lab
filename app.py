import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, FileText, Trash2, BookOpen } from 'lucide-react';

const RAGChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [context, setContext] = useState('');
  const [showContext, setShowContext] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Sample knowledge base - in a real app, this would be from uploaded documents
  const knowledgeBase = `
    Claude is an AI assistant created by Anthropic. Claude is helpful, harmless, and honest.
    
    Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei, along with several other former members of OpenAI.
    The company focuses on AI safety and research.
    
    Claude can help with a wide variety of tasks including writing, analysis, math, coding, and creative projects.
    Claude is designed to be helpful, harmless, and honest in all interactions.
    
    The latest version of Claude is part of the Claude 3 model family, which includes Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku.
    These models offer different balances of intelligence, speed, and cost.
  `;

  // Simple keyword-based retrieval (mimics vector similarity)
  const retrieveRelevantContext = (query) => {
    const sentences = knowledgeBase.split('\n').filter(s => s.trim());
    const queryWords = query.toLowerCase().split(' ');
    
    // Score each sentence based on keyword matches
    const scoredSentences = sentences.map(sentence => {
      const sentenceLower = sentence.toLowerCase();
      const score = queryWords.reduce((acc, word) => {
        return acc + (sentenceLower.includes(word) ? 1 : 0);
      }, 0);
      return { sentence, score };
    });

    // Get top 3 relevant sentences
    const topSentences = scoredSentences
      .filter(s => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(s => s.sentence);

    return topSentences.join('\n');
  };

  // Simulate API call to Hugging Face model
  const generateResponse = async (userMessage, relevantContext) => {
    // In a real implementation, you would call Hugging Face API here
    // For demo purposes, we'll create a rule-based response
    
    await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay

    const messageLower = userMessage.toLowerCase();
    
    if (!relevantContext) {
      return "I don't have enough information in my knowledge base to answer that question. Please try asking something else or add more context to my knowledge base.";
    }

    // Create a contextual response based on retrieved information
    let response = "Based on the information I have:\n\n";
    
    if (messageLower.includes('who') || messageLower.includes('what is claude')) {
      response += "Claude is an AI assistant created by Anthropic, designed to be helpful, harmless, and honest. ";
    } else if (messageLower.includes('anthropic')) {
      response += "Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei. The company focuses on AI safety and research. ";
    } else if (messageLower.includes('help') || messageLower.includes('do')) {
      response += "Claude can help with various tasks including writing, analysis, math, coding, and creative projects. ";
    } else if (messageLower.includes('version') || messageLower.includes('model')) {
      response += "The latest Claude models are part of the Claude 3 family, which includes Opus, Sonnet, and Haiku variants. ";
    } else {
      response += relevantContext.split('\n')[0] + " ";
    }
    
    response += "\n\nIs there anything specific you'd like to know more about?";
    
    return response;
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    // Add user message
    const newMessages = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);

    try {
      // Retrieve relevant context using RAG
      const relevantContext = retrieveRelevantContext(userMessage);
      setContext(relevantContext);

      // Generate response using the context
      const response = await generateResponse(userMessage, relevantContext);

      // Add assistant message
      setMessages([...newMessages, { role: 'assistant', content: response }]);
    } catch (error) {
      setMessages([...newMessages, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setContext('');
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-md border-b border-gray-200 p-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <BookOpen className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">RAG Chatbot</h1>
              <p className="text-sm text-gray-600">Powered by Retrieval-Augmented Generation</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowContext(!showContext)}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 transition-colors"
            >
              <FileText size={18} />
              {showContext ? 'Hide' : 'Show'} Context
            </button>
            <button
              onClick={clearChat}
              className="flex items-center gap-2 px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
            >
              <Trash2 size={18} />
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex max-w-4xl w-full mx-auto gap-4 p-4">
        {/* Chat Area */}
        <div className={`flex flex-col bg-white rounded-xl shadow-lg overflow-hidden transition-all ${showContext ? 'w-2/3' : 'w-full'}`}>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-4">
                  <div className="bg-indigo-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto">
                    <BookOpen className="text-indigo-600" size={32} />
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-700">Welcome to RAG Chatbot!</h2>
                  <p className="text-gray-500 max-w-md">
                    This chatbot uses Retrieval-Augmented Generation to answer questions based on a knowledge base.
                    Try asking about Claude or Anthropic!
                  </p>
                  <div className="bg-gray-50 p-4 rounded-lg max-w-md mx-auto text-left">
                    <p className="text-sm font-semibold text-gray-700 mb-2">Try asking:</p>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• What is Claude?</li>
                      <li>• Who founded Anthropic?</li>
                      <li>• What can Claude help with?</li>
                      <li>• What are the Claude 3 models?</li>
                    </ul>
                  </div>
                </div>
              </div>
            ) : (
              <>
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                        message.role === 'user'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 rounded-2xl px-4 py-3 flex items-center gap-2">
                      <Loader2 className="animate-spin text-indigo-600" size={18} />
                      <span className="text-gray-600">Thinking...</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4 bg-gray-50">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about Claude or Anthropic..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                {loading ? (
                  <Loader2 className="animate-spin" size={20} />
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Context Panel */}
        {showContext && (
          <div className="w-1/3 bg-white rounded-xl shadow-lg p-6 overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <FileText size={20} className="text-indigo-600" />
              Retrieved Context
            </h3>
            {context ? (
              <div className="bg-indigo-50 p-4 rounded-lg">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{context}</p>
              </div>
            ) : (
              <div className="text-center text-gray-400 py-8">
                <FileText size={48} className="mx-auto mb-2 opacity-50" />
                <p className="text-sm">No context retrieved yet</p>
                <p className="text-xs mt-1">Ask a question to see relevant information</p>
              </div>
            )}
            
            <div className="mt-6 pt-6 border-t border-gray-200">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Knowledge Base Preview</h4>
              <div className="bg-gray-50 p-3 rounded text-xs text-gray-600 max-h-40 overflow-y-auto">
                {knowledgeBase.trim().split('\n').slice(0, 5).join('\n')}...
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RAGChatbot;
