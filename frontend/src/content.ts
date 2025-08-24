// Centralized UI copy for AI Content Detector Pro
// This mirrors the requested structure and can be localized later.

const UI_TEXT = {
  app: {
    title: "AI Content Detector Pro",
    subtitle:
      "Analyze text or documents to estimate whether content is human‑written or AI‑generated using multiple complementary methods.",
    badges: {
      beta: "Beta",
      privacy_first: "Privacy‑First",
      no_login: "No Login Required",
    },
    footer_note:
      "Results are probabilistic estimates based on statistical and machine learning methods. Use judgment and additional context for high‑stakes decisions.",
  },

  sidebar: {
    title: "Analysis Controls",
    sections: {
      methods: "Choose Detection Methods",
      weights: "Method Weights",
      thresholds: "Thresholds",
      display: "Display Options",
      links: "Documentation",
    },
    method_toggles: {
      stylometric: "Stylometric Analysis",
      perplexity: "Perplexity Analysis",
      ml: "ML Classification",
      combined: "Combined (Recommended)",
    },
    weights: {
      stylometric: "Stylometric Weight",
      perplexity: "Perplexity Weight",
      ml: "ML Weight",
      reset: "Reset Weights",
      hint: "Adjust how much each method contributes to the final score.",
    },
    thresholds: {
      human_cutoff: "Human Verdict Threshold",
      ai_cutoff: "AI Verdict Threshold",
      calibrated: "Use Calibrated Thresholds",
      reset: "Reset Thresholds",
      hint: "Scores above the human threshold are labeled human; below the AI threshold are labeled AI; otherwise inconclusive.",
    },
    display_options: {
      realtime: "Real‑time analysis",
      explanations: "Show detailed explanations",
      visuals: "Show interactive charts",
      sentence_view: "Show sentence‑level breakdown",
      download_report: "Enable report export",
    },
    links: {
      docs: "Documentation",
      source: "Source Code",
      issues: "Report an Issue",
      support: "Support",
    },
    buttons: { run: "Run Analysis", reset: "Reset" },
  },

  input: {
    header: "Provide Text or Upload a File",
    text_label: "Paste your text",
    text_placeholder: "Paste or type your content here…",
    word_count: "Word count: {count}",
    upload_label: "Upload a document",
    upload_hint: "Accepted: .txt, .docx, .pdf • Max 10 MB • Text only.",
    choose_file: "Choose File",
    replace_file: "Replace File",
    remove_file: "Remove File",
    parsing_info: "We extract text locally in your browser/server session.",
    samples: {
      header: "Try a Sample",
      human: "Sample: Human‑like review",
      ai: "Sample: Academic AI draft",
    },
  },

  methods: {
    header: "Choose Detection Methods",
    stylometric: "Stylometric Analysis",
    perplexity: "Perplexity Analysis",
    ml: "ML Classification",
    combined: "Combined (Recommended)",
  },

  method_help: {
    stylometric:
      "Measures writing style patterns such as sentence length, punctuation, vocabulary variety, and capitalization to infer typical human variability.",
    perplexity:
      "Estimates how predictable the text is to a language model. Lower perplexity can mean more typical phrasing; higher may indicate unusual or templated text.",
    ml: "Uses a trained classifier on TF‑IDF features and other signals to predict AI vs. human probability.",
    combined:
      "Blends the methods using adjustable weights to provide a robust final estimate.",
  },

  controls: {
    weights: {
      title: "Weights",
      stylometric: "Stylometric",
      perplexity: "Perplexity",
      ml: "ML",
      reset: "Reset Weights to Default",
    },
    thresholds: {
      title: "Thresholds",
      human: "Human label ≥",
      ai: "AI label ≤",
      reset: "Reset Thresholds",
    },
  },

  actions: {
    analyze: "Analyze",
    analyzing: "Analyzing…",
    cancel: "Cancel",
    progress: {
      preparing: "Preparing models…",
      vectorizing: "Vectorizing text…",
      scoring: "Scoring with selected methods…",
      combining: "Combining results…",
      finalizing: "Finalizing output…",
    },
  },

  results: {
    header: "Results",
    final_label_human: "Likely Human‑Written",
    final_label_ai: "Likely AI‑Generated",
    final_label_uncertain: "Inconclusive / Mixed Signals",
    confidence: "Confidence: {pct}%",
    timestamp: "Generated at {timestamp}",
    explanatory_badge: "Weighted blend of selected methods",
  },

  breakdown: {
    header: "Method Breakdown",
    cards: {
      stylometric: {
        title: "Stylometric Score",
        findings: [
          "Avg sentence length: {value}",
          "Vocabulary richness: {value}",
          "Punctuation ratio: {value}",
        ],
      },
      perplexity: {
        title: "Perplexity Score",
        findings: [
          "Raw perplexity: {ppl}",
          "Burstiness (variance): {burst}",
          "Normalized score: {score}",
        ],
      },
      ml: {
        title: "ML Classification",
        findings: [
          "AI probability: {ai_pct}%",
          "Human probability: {human_pct}%",
          "Threshold: {threshold}",
        ],
      },
      combined: {
        title: "Combined Score",
        findings: [
          "Weights — Stylometric: {w_sty}, Perplexity: {w_ppl}, ML: {w_ml}",
          "Score consistency: {consistency}",
        ],
      },
    },
  },

  visualizations: {
    header: "Visualizations",
    charts: {
      feature_importance: "Top Features (ML Importance)",
      sentence_length: "Sentence Length Distribution",
      vocab_richness: "Vocabulary Richness Over Text",
      punctuation_use: "Punctuation Usage",
      capitalization: "Capitalization Ratio",
      perplexity_gauge: "Perplexity Gauge",
      probability_pie: "Human vs. AI Probability",
    },
  },

  explanations: {
    header: "Explanations",
    expand_all: "Expand All",
    collapse_all: "Collapse All",
    stylometric:
      "Human authors tend to vary sentence length and vocabulary more. We compute features (e.g., variance in sentence length, unique word ratio, punctuation ratio) and convert them into a score.",
    perplexity:
      "Perplexity reflects how ‘expected’ the text is under a language model. We include a burstiness measure to capture human‑like variability across sentences.",
    ml: "The ML classifier uses TF‑IDF n‑grams and logistic regression. The probability indicates how similar the text is to the model’s AI‑generated samples.",
    combined:
      "The final score is a weighted blend of method scores with optional thresholding to produce Human/AI/Inconclusive labels.",
  },

  file_handling: {
    select: "Select a file",
    replace: "Replace file",
    remove: "Remove file",
    parsing: "Parsing file…",
    parsed: "File parsed successfully: {name}",
    accepted_types: "Accepted: .txt, .docx, .pdf",
    size_limit: "Max size: 10 MB",
  },

  empty_states: {
    no_input: "No text provided. Paste text or upload a document to begin.",
    no_methods: "No methods selected. Enable at least one analysis method.",
    text_too_short: "Text is too short to analyze. Add more content.",
  },

  toasts: {
    success: "Analysis complete.",
    warning: "Mixed signals detected — consider reviewing explanations.",
    info: "Using default weights and thresholds.",
    copied: "Copied to clipboard.",
  },

  errors: {
    parsing_failed: "Could not parse the file. Please try a different file or format.",
    unsupported_file: "Unsupported file type. Allowed: .txt, .docx, .pdf.",
    size_limit: "File exceeds the 10 MB size limit.",
    short_text: "The text is too short for reliable analysis.",
    model_load: "Model could not be loaded. Please retry.",
    timeout: "Analysis timed out. Try again or analyze a shorter text.",
    network: "Network error. Check connection and retry.",
  },

  privacy_ethics: {
    disclaimer:
      "This tool provides probabilistic signals and can be affected by paraphrasing, editing, and domain shifts. Do not use it as the sole basis for punitive actions.",
    no_third_party:
      "We do not sell or share your content with third parties. Data is processed for analysis only.",
  },

  footer_links: {
    docs: "Documentation",
    source: "Source Code",
    issues: "Issues",
    support: "Support",
  },

  buttons_short: {
    analyze: "Analyze",
    cancel: "Cancel",
    copy: "Copy",
    download: "Download",
    expand: "Expand",
    collapse: "Collapse",
  },

  export: {
    report_title: "AI Content Detector Pro — Analysis Report",
    sections: {
      overview: "Overview",
      final_verdict: "Final Verdict",
      method_breakdown: "Method Breakdown",
      visualizations: "Visualizations",
      explanations: "Explanations",
      appendix: "Appendix",
    },
    disclaimer:
      "This report reflects probabilistic assessments produced by AI models and statistical methods. Interpret with caution and use additional context where necessary.",
  },
} as const;

export default UI_TEXT;


