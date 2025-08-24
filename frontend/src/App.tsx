import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'
import { motion, AnimatePresence } from 'framer-motion'
import { Switch } from '@headlessui/react'
import clsx from 'clsx'
import UI_TEXT from './content'

type Method = 'stylometric' | 'perplexity' | 'ml' | 'combined'

const API_BASE = 'http://localhost:8000'

export default function App() {
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [fileName, setFileName] = useState('')
  const [methods, setMethods] = useState<Record<Method, boolean>>({
    stylometric: true,
    perplexity: true,
    ml: true,
    combined: true
  })
  const [weights, setWeights] = useState({
    stylometric: 0.4,
    perplexity: 0.3,
    ml: 0.3
  })
  const [thresholds, setThresholds] = useState({ human: 0.6, ai: 0.4 })
  const [displayOptions, setDisplayOptions] = useState({
    explanations: true,
    visuals: true,
    sentence_view: false,
    download_report: true
  })
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [results, setResults] = useState<Record<string, any> | null>(null)
  const [expandedMethods, setExpandedMethods] = useState<Record<string, boolean>>({})
  const [dark, setDark] = useState(false)
  const [toasts, setToasts] = useState<Array<{id: string, type: 'success' | 'warning' | 'info' | 'error', message: string}>>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const root = document.documentElement
    if (dark) {
      root.classList.add('dark')
      localStorage.setItem('theme-dark', '1')
    } else {
      root.classList.remove('dark')
      localStorage.setItem('theme-dark', '0')
    }
  }, [dark])

  const addToast = (type: 'success' | 'warning' | 'info' | 'error', message: string) => {
    const id = Math.random().toString(36)
    setToasts(prev => [...prev, { id, type, message }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (file.size > 10 * 1024 * 1024) {
      addToast('error', 'File too large. Maximum size is 10MB.')
      return
    }

    if (!['.txt', '.docx', '.pdf'].some(ext => file.name.toLowerCase().endsWith(ext))) {
      addToast('error', 'Unsupported file type. Please use .txt, .docx, or .pdf files.')
      return
    }

    setFile(file)
    setFileName(file.name)
    addToast('info', `File uploaded: ${file.name}`)
  }

  const removeFile = () => {
    setFile(null)
    setFileName('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleAnalyze = async () => {
    if (!text.trim() && !file) {
      addToast('warning', 'Please provide text or upload a file to analyze.')
      return
    }

    const selectedMethods = Object.entries(methods).filter(([_, enabled]) => enabled).map(([method]) => method)
    if (selectedMethods.length === 0) {
      addToast('warning', 'Please select at least one analysis method.')
      return
    }

    let analysisText = ''
    if (file) {
      try {
        const fileContent = await file.text()
        analysisText = fileContent.trim()
        if (!analysisText) {
          addToast('error', 'Could not extract text from the uploaded file.')
          return
        }
      } catch (error) {
        addToast('error', 'Error reading file content')
        return
      }
    } else {
      analysisText = text.trim()
    }

    if (analysisText.split(' ').length < 10) {
      addToast('warning', 'Text must be at least 10 words long.')
      return
    }

    setLoading(true)
    setProgress('Preparing analysis...')
    
    try {
      const methodMapping: Record<string, string> = {
        'stylometric': 'Stylometric Analysis',
        'perplexity': 'Perplexity Analysis', 
        'ml': 'ML Classification',
        'combined': 'Combined Analysis'
      }
      
      const calls = selectedMethods.map(method => 
        axios.post(`${API_BASE}/analyze`, { 
          text: analysisText, 
          method: methodMapping[method] || method 
        })
      )
      
      setProgress('Analyzing text...')
      const responses = await Promise.all(calls)
      
      console.log('Raw API responses:', responses)
      
      const combinedResults = responses.reduce((acc, res, idx) => {
        const method = selectedMethods[idx]
        console.log(`Processing method ${method}:`, res.data)
        return { ...acc, [method]: res.data }
      }, {})
      
      console.log('Combined results:', combinedResults)
      setResults(combinedResults)
      addToast('success', 'Analysis completed successfully!')
    } catch (error) {
      console.error('Analysis failed:', error)
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          addToast('error', 'Cannot connect to the backend server. Please make sure the API is running on port 8000.')
        } else if (error.response) {
          addToast('error', `Backend error: ${error.response.status}`)
        } else {
          addToast('error', 'Network error. Please check your connection.')
        }
      } else {
        addToast('error', 'Analysis failed. Please try again.')
      }
    } finally {
      setLoading(false)
      setProgress('')
    }
  }

  const getFinalVerdict = () => {
    if (!results) return { label: '', confidence: 0, color: '', score: 0.5 }
    
    console.log('Raw results:', results)
    
    let combinedScore = 0.5
    let totalWeight = 0
    
    const resultsData = results as Record<string, any>
    // Unwrap extra nesting if present
    const stylometricData = resultsData.stylometric?.stylometric || resultsData.stylometric
    const perplexityData = resultsData.perplexity?.perplexity || resultsData.perplexity
    const mlData = resultsData.ml?.ml || resultsData.ml

    if (stylometricData?.stylometric_score !== undefined) {
      const stylometricScore = stylometricData.stylometric_score
      console.log('Stylometric score:', stylometricScore)
      combinedScore += stylometricScore * 0.25
      totalWeight += 0.25
    }
    if (perplexityData?.perplexity !== undefined) {
      const perplexityScore = perplexityData.perplexity
      console.log('Perplexity score:', perplexityScore)
      combinedScore += perplexityScore * 0.35
      totalWeight += 0.35
    }
    if (mlData?.ml_score !== undefined) {
      const mlScore = mlData.ml_score
      console.log('ML score:', mlScore)
      const humanProbability = 1 - mlScore
      combinedScore += humanProbability * 0.40
      totalWeight += 0.40
    }
    if (totalWeight > 0) {
      combinedScore = combinedScore / totalWeight
    }
    console.log('Combined score:', combinedScore)
    const confidence = Math.abs(combinedScore - 0.5) * 2 * 100
    console.log('Confidence:', confidence)
    console.log('Thresholds:', thresholds)
    if (combinedScore >= thresholds.human) {
      return { 
        label: 'Likely Human', 
        confidence: Math.min(confidence, 100), 
        color: 'text-emerald-600',
        score: combinedScore
      }
    } else if (combinedScore <= thresholds.ai) {
      return { 
        label: 'Likely AI', 
        confidence: Math.min(confidence, 100), 
        color: 'text-rose-600',
        score: combinedScore
      }
    } else {
      return { 
        label: 'Uncertain', 
        confidence: Math.min(confidence, 100), 
        color: 'text-amber-600',
        score: combinedScore
      }
    }
  }

  const verdict = getFinalVerdict()
  const wordCount = file ? 'File uploaded' : text.trim().split(/\s+/).filter(word => word.length > 0).length

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="sticky top-0 z-20 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200 dark:border-slate-700">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
          <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  AI Content Detector Pro
                </h1>
                <p className="text-slate-600 dark:text-slate-300 text-sm mt-1">
                  Advanced AI-generated content detection using multiple analysis techniques
                </p>
              </div>
              <div className="flex space-x-2">
                <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 rounded-full">
                  Beta
                </span>
                <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 rounded-full">
                  Privacy First
                </span>
              </div>
          </div>
            <button 
              onClick={() => setDark(d => !d)} 
              className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
            >
              {dark ? '🌞' : '🌙'}
          </button>
          </div>
        </div>
      </header>

      <div className="flex max-w-7xl mx-auto">
        {/* Sidebar */}
        <aside className="w-80 h-screen bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl border-r border-slate-200 dark:border-slate-700 shadow-lg sticky top-0">
          <div className="p-4 space-y-6 sidebar-scroll h-full overflow-y-auto">
            {/* Header */}
            <div className="border-b border-slate-200 dark:border-slate-700 pb-4">
              <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100 mb-2">
                Analysis Settings
              </h3>
              <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                Configure your analysis settings and methods
              </p>
            </div>
            
            {/* Method Selection */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100 uppercase tracking-wide">
                  Methods
                </h4>
                <span className="text-xs text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded-full">
                  {Object.values(methods).filter(Boolean).length}/4
                </span>
              </div>
              
              <div className="space-y-2">
                {Object.entries(methods).map(([method, enabled]) => (
                  <div key={method} className="group">
                    <div className="flex items-start space-x-3 p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                      <Switch
                        checked={enabled}
                        onChange={(checked) => setMethods(prev => ({ ...prev, [method]: checked }))}
                        className={`${
                          enabled ? 'bg-blue-600' : 'bg-slate-200 dark:bg-slate-700'
                        } relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 flex-shrink-0 mt-0.5`}
                      >
                        <span className={`${
                          enabled ? 'translate-x-4' : 'translate-x-1'
                        } inline-block h-3 w-3 transform rounded-full bg-white transition-transform`} />
                      </Switch>
                      
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-medium text-slate-900 dark:text-slate-100 block capitalize">
                          {method}
                        </span>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                          {method === 'stylometric' && 'Analyzes writing style patterns'}
                          {method === 'perplexity' && 'Measures text complexity and naturalness'}
                          {method === 'ml' && 'Machine learning-based detection'}
                          {method === 'combined' && 'Combines all methods for best accuracy'}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Stats */}
            <div className="pt-3 border-t border-slate-200 dark:border-slate-700">
              <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-3">
                <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100 mb-2 uppercase tracking-wide">
                  Quick Stats
                </h4>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600 dark:text-slate-400">Methods Active:</span>
                    <span className="font-medium text-slate-900 dark:text-slate-100 bg-blue-100 dark:bg-blue-900/30 px-2 py-0.5 rounded text-xs">
                      {Object.values(methods).filter(Boolean).length}/4
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6 space-y-6">
          {/* Input Section */}
          <motion.section 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="pro-card p-8"
          >
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                Input Text or Upload File
              </h2>
              <p className="text-slate-600 dark:text-slate-400">
                Upload a document or paste your text to begin analysis
              </p>
            </div>
            
            {/* File Upload */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 uppercase tracking-wide">
                Upload File
              </label>
              <div className="flex items-center space-x-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.docx,.pdf"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-secondary hover-lift"
                >
                  📁 Choose File
                </button>
                {file && (
                  <>
                    <div className="flex items-center space-x-2 bg-green-50 dark:bg-green-900/20 px-3 py-2 rounded-lg border border-green-200 dark:border-green-800">
                      <span className="text-green-600 dark:text-green-400">✓</span>
                      <span className="text-sm text-green-700 dark:text-green-300 font-medium">{fileName}</span>
                    </div>
                    <button
                      onClick={removeFile}
                      className="text-sm text-red-600 hover:text-red-700 dark:text-red-400 hover:underline"
                    >
                      Remove
                    </button>
                  </>
                )}
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                Supported formats: .txt, .docx, .pdf (max 10MB)
              </p>
            </div>

            {/* Text Input */}
            <div className="mb-8">
              <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 uppercase tracking-wide">
                Or Enter Text
              </label>
          <textarea
                className="input-field h-48 resize-none"
                placeholder="Enter the text you want to analyze for AI-generated content detection..."
            value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <div className="flex justify-between items-center mt-3">
                <span className="text-sm text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-3 py-1 rounded-full">
                  Words: {wordCount}
                </span>
                <div className="flex space-x-3">
                  <button
                    onClick={() => setText("This is a sample human-written text that demonstrates natural language patterns and variability in sentence structure. It includes various punctuation marks, different sentence lengths, and natural flow that humans typically produce when writing.")}
                    className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 hover:underline bg-blue-50 dark:bg-blue-900/20 px-3 py-1 rounded-lg transition-colors"
                  >
                    Sample Human
                  </button>
                  <button
                    onClick={() => setText("The implementation of artificial intelligence systems requires careful consideration of multiple factors including algorithmic complexity, computational efficiency, and ethical implications. Such systems must be designed with robust error handling mechanisms and comprehensive testing protocols to ensure reliability and safety in real-world applications.")}
                    className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 hover:underline bg-blue-50 dark:bg-blue-900/20 px-3 py-1 rounded-lg transition-colors"
                  >
                    Sample AI
                  </button>
                </div>
              </div>
            </div>

            {/* Analyze Button */}
            <div className="flex justify-center">
              <button
                onClick={handleAnalyze}
                disabled={loading || (!text.trim() && !file)}
                className={clsx(
                  'px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-blue-500 focus:ring-offset-2',
                  loading || (!text.trim() && !file)
                    ? 'bg-slate-400 cursor-not-allowed shadow-none'
                    : 'bg-gradient-to-r from-blue-600 via-purple-600 to-blue-700 text-white shadow-xl hover:shadow-2xl hover:from-blue-700 hover:via-purple-700 hover:to-blue-800'
                )}
              >
                {loading ? (
                  <div className="flex items-center space-x-3">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Analyzing...</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <span>🔍</span>
                    <span>Analyze Content</span>
                  </div>
                )}
              </button>
            </div>

            {/* Progress */}
            {loading && progress && (
              <div className="mt-4 text-center">
                <p className="text-sm text-slate-600 dark:text-slate-400">{progress}</p>
          </div>
            )}
        </motion.section>

          {/* Results Section */}
          <AnimatePresence>
            {results && (
              <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Final Verdict */}
                <div className="pro-card p-8 text-center bg-gradient-to-br from-white to-slate-50 dark:from-slate-800 dark:to-slate-900 border border-slate-200 dark:border-slate-700">
                  <div className="mb-8">
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                      Analysis Results
                    </h2>
                    <div className="w-32 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-blue-600 mx-auto rounded-full shadow-lg"></div>
                  </div>
                  <div className="mb-8">
                    <div className="inline-flex items-center justify-center w-32 h-32 rounded-full bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-700 dark:to-slate-800 mb-6 shadow-lg">
                      <h3 className={`text-5xl font-bold ${verdict.color}`}>{verdict.label === 'Likely Human' ? '👤' : verdict.label === 'Likely AI' ? '🤖' : '❓'}</h3>
                    </div>
                    <div className={`text-3xl font-bold ${verdict.color} mb-2`}>{verdict.label}</div>
                    <div className="text-lg text-slate-700 dark:text-slate-300 mb-2">{Math.round(verdict.confidence)}%<span className="ml-2 text-base font-medium text-slate-500">CONFIDENCE LEVEL</span></div>
                  </div>
                  {/* Probability Pie Chart */}
                  <div className="max-w-xs mx-auto mb-8">
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={[{ name: 'Human', value: verdict.score }, { name: 'AI', value: 1 - verdict.score }]}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={70}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        >
                          <Cell key="human" fill="#10b981" />
                          <Cell key="ai" fill="#ef4444" />
                        </Pie>
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  {/* Method Scores Bar Chart */}
                  <div className="max-w-lg mx-auto mb-8">
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={[
                        { name: 'Stylometric', value: (results?.stylometric?.stylometric?.stylometric_score || results?.stylometric?.stylometric_score || 0.5) },
                        { name: 'Perplexity', value: (results?.perplexity?.perplexity?.perplexity || results?.perplexity?.perplexity || 0.5) },
                        { name: 'ML', value: (results?.ml?.ml?.ml_score !== undefined ? 1 - results.ml.ml.ml_score : (results?.ml?.ml_score !== undefined ? 1 - results.ml.ml_score : 0.5)) },
                        { name: 'Combined', value: (results?.combined?.combined?.combined_score || results?.combined?.combined_score || verdict.score) }
                      ]}>
                        <XAxis dataKey="name" />
                        <YAxis domain={[0, 1]} tickFormatter={v => `${Math.round(v * 100)}%`} />
                        <Tooltip formatter={v => `${Math.round((v as number) * 100)}%`} />
                        <Bar dataKey="value" fill="#6366f1" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  {/* Final Result Summary Card */}
                  <div className="pro-card p-4 mt-4 bg-gradient-to-br from-green-50 to-white dark:from-emerald-900/30 dark:to-slate-900 border border-emerald-200 dark:border-emerald-700 text-center">
                    <div className="text-lg font-semibold mb-2">Summary</div>
                    <div className="text-base text-slate-700 dark:text-slate-200 mb-1">Final verdict: <span className={`font-bold ${verdict.color}`}>{verdict.label}</span></div>
                    <div className="text-base text-slate-700 dark:text-slate-200 mb-1">Confidence: <span className="font-bold">{Math.round(verdict.confidence)}%</span></div>
                    <div className="text-base text-slate-700 dark:text-slate-200">This result is based on a weighted combination of stylometric, perplexity, and ML analysis. For best accuracy, use longer and more natural text samples.</div>
                  </div>
                </div>

        {/* Method Breakdown */}
                {displayOptions.explanations && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                        Method Breakdown
                      </h3>
                      <p className="text-slate-600 dark:text-slate-400">
                        Detailed analysis from each detection method
                      </p>
                    </div>
                    
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {Object.entries(results).map(([method, data]) => {
                        // Unwrap extra nesting if present
                        let methodData = (data as any)[method] || data
                        let score = 0.5
                        let scoreLabel = 'Confidence Score'
                        
                        if (method === 'stylometric' && methodData.stylometric_score !== undefined) {
                          score = methodData.stylometric_score
                          console.log(`Stylometric score: ${score}`)
                        } else if (method === 'perplexity' && methodData.perplexity !== undefined) {
                          score = methodData.perplexity
                          console.log(`Perplexity score: ${score}`)
                        } else if (method === 'ml' && methodData.ml_score !== undefined) {
                          score = 1 - methodData.ml_score
                          console.log(`ML score: ${methodData.ml_score}, inverted: ${score}`)
                        } else if (method === 'combined' && methodData.combined_score !== undefined) {
                          score = methodData.combined_score
                          console.log(`Combined score: ${score}`)
                        }
                        
                        console.log(`Final score for ${method}: ${score}`)
                        
                        return (
                          <div
                            key={method}
                            className="pro-card p-6 pro-card-hover bg-white dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700"
                          >
                            <div className="flex items-center justify-between mb-6">
                              <h3 className="text-lg font-semibold capitalize text-slate-900 dark:text-slate-100">
                                {method}
                              </h3>
                              <button
                                onClick={() => setExpandedMethods(prev => ({ ...prev, [method]: !prev[method] }))}
                                className="text-blue-600 hover:text-blue-700 dark:text-blue-400 text-sm font-medium hover:underline transition-colors px-3 py-1 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20"
                              >
                                {expandedMethods[method] ? 'Collapse' : 'Expand'}
                              </button>
                            </div>
                            
                            <div className="space-y-6">
                              <div className="text-center">
                                <div className="text-5xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                                  {Math.round(score * 100)}%
                                </div>
                                <div className="text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wide font-medium">
                                  {scoreLabel}
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      })}
            </div>
            </div>
                )}

          </motion.section>
        )}
          </AnimatePresence>

          {/* Privacy Notice */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>Privacy Notice:</strong> Your text is analyzed locally and not shared with third parties. Results are probabilistic and should be used as guidance only.
            </p>
          </div>
      </main>
      </div>

      {/* Footer */}
      <footer className="bg-white/60 dark:bg-slate-900/60 backdrop-blur-xl border-t border-slate-200 dark:border-slate-700 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="text-center">
            <p className="text-sm text-slate-500 dark:text-slate-400">
              © 2024 AI Content Detector Pro. Advanced AI-generated content detection using multiple analysis techniques.
            </p>
          </div>
        </div>
      </footer>

      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        <AnimatePresence>
          {toasts.map((toast) => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              className={clsx(
                'px-4 py-3 rounded-lg shadow-lg text-white text-sm font-medium min-w-80',
                toast.type === 'success' && 'bg-green-500',
                toast.type === 'warning' && 'bg-yellow-500',
                toast.type === 'info' && 'bg-blue-500',
                toast.type === 'error' && 'bg-red-500'
              )}
            >
              {toast.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  )
}
