import os
import math
import pickle
import numpy as np
import nltk
import re
from collections import Counter
from typing import Dict, List, Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from nltk.corpus import brown
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends


# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')


class AIContentDetectorCore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True
        )
        self.classifier = LogisticRegression(
            solver='liblinear', random_state=42, max_iter=1000, class_weight='balanced'
        )
        self.model_path = "ai_detector_model.pkl"
        self.threshold = 0.5
        self.metrics: Dict[str, Any] = {}
        self._lm = None
        self.load_or_train_model()

    def load_or_train_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.classifier = model_data['classifier']
                    self.threshold = model_data.get('threshold', 0.5)
                    self.metrics = model_data.get('metrics', {})
                return
            except Exception:
                pass
        self.train_model()

    # Synthetic data generation (same as Streamlit app)
    def generate_human_like_texts(self, num_samples: int) -> List[str]:
        human_patterns = [
            "I think that this is really interesting because it shows how people naturally write with varying sentence lengths and sometimes make small mistakes or use informal language.",
            "The weather today was quite unpredictable. It started sunny but then clouds rolled in and it began to rain. I forgot my umbrella at home, which was pretty annoying.",
            "My friend told me about this new restaurant downtown. The food was amazing, though a bit expensive. We had to wait like 20 minutes for a table, but it was totally worth it.",
            "I'm not sure if I should take that job offer. The salary is good, but the commute would be really long. Plus, I'd have to move to a different city.",
            "Yesterday I went to the store to buy some groceries. I got bread, milk, eggs, and some vegetables. The cashier was really friendly and we chatted for a bit.",
            "This book is really hard to put down. The characters are so well-developed and the plot has so many unexpected twists. I stayed up until 2 AM reading it.",
            "I tried cooking that new recipe I found online. It didn't turn out exactly like the picture, but it still tasted pretty good. I think I used too much salt though.",
            "The movie was okay, I guess. The special effects were cool, but the story was kind of predictable. My friend loved it though, so maybe it's just me.",
            "I need to remember to call the dentist tomorrow to schedule my appointment. I've been putting it off for weeks now, which is probably not a good idea.",
            "That meeting was so boring. The presenter just kept talking and talking without really saying anything important. I almost fell asleep."
        ]
        texts = []
        for _ in range(num_samples):
            base_text = np.random.choice(human_patterns)
            if np.random.random() > 0.5:
                base_text += " " + np.random.choice([
                    "Anyway, that's what happened.",
                    "I'm not really sure about that.",
                    "What do you think?",
                    "It was kind of weird.",
                    "I guess we'll see."
                ])
            texts.append(base_text)
        return texts

    def generate_ai_like_texts(self, num_samples: int) -> List[str]:
        ai_patterns = [
            "The implementation of this system demonstrates a comprehensive approach to data analysis, incorporating multiple methodologies to ensure accurate results.",
            "Furthermore, the analysis reveals significant correlations between the variables under investigation, suggesting a robust relationship that warrants further examination.",
            "In conclusion, the findings presented herein provide compelling evidence for the hypothesis, supported by rigorous statistical analysis and peer-reviewed research.",
            "The methodology employed in this study follows established protocols and best practices, ensuring reproducibility and reliability of the experimental outcomes.",
            "Additionally, the results indicate a clear pattern of behavior that aligns with theoretical predictions, thereby validating the underlying assumptions of the model.",
            "The comprehensive review of the literature reveals a consensus among researchers regarding the fundamental principles governing this phenomenon.",
            "Moreover, the data analysis demonstrates statistical significance across multiple dimensions, providing strong support for the proposed framework.",
            "The systematic approach adopted in this investigation ensures methodological rigor while maintaining practical applicability in real-world scenarios.",
            "Furthermore, the empirical evidence presented herein contributes to the growing body of knowledge in this field, offering valuable insights for future research.",
            "The theoretical framework provides a solid foundation for understanding the complex interactions between various factors influencing the observed outcomes."
        ]
        texts = []
        for _ in range(num_samples):
            base_text = np.random.choice(ai_patterns)
            if np.random.random() > 0.5:
                base_text += " " + np.random.choice([
                    "This conclusion is supported by extensive empirical evidence.",
                    "Further research is warranted to explore these findings.",
                    "The implications of these results are far-reaching.",
                    "This analysis provides valuable insights for practitioners.",
                    "The methodology ensures robust and reliable outcomes."
                ])
            texts.append(base_text)
        return texts

    def train_model(self, use_real_data: bool = False):
        texts = self.generate_human_like_texts(300) + self.generate_ai_like_texts(300)
        labels = [0] * 300 + [1] * 300
        X_train_texts, X_val_texts, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train = self.vectorizer.fit_transform(X_train_texts)
        X_val = self.vectorizer.transform(X_val_texts)
        self.classifier.fit(X_train, y_train)
        val_proba = self.classifier.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, val_proba)
        youden_j = tpr - fpr
        best_idx = int(youden_j.argmax())
        self.threshold = float(thresholds[best_idx])
        acc = float(accuracy_score(y_val, (val_proba >= self.threshold).astype(int)))
        auc = float(roc_auc_score(y_val, val_proba))
        self.metrics = {'val_accuracy': acc, 'val_auc': auc, 'threshold': self.threshold}
        with open(self.model_path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'classifier': self.classifier,
                        'threshold': self.threshold, 'metrics': self.metrics}, f)


class AnalyzerCore:
    def __init__(self):
        self.detector = AIContentDetectorCore()
        self._lm = None

    def calculate_stylometric_features(self, text: str) -> Dict[str, float]:
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        if not words or not sentences:
            return {}
        avg_sentence_length = float(np.mean([len(nltk.word_tokenize(s)) for s in sentences]))
        vocab_richness = len(set(words)) / len(words)
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(10)
        common_words_ratio = sum(freq for _, freq in most_common_words) / len(words)
        sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
        sentence_length_variance = float(np.var(sentence_lengths))
        word_lengths = [len(w) for w in words]
        avg_word_length = float(np.mean(word_lengths))
        word_length_variance = float(np.var(word_lengths))
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        punctuation_ratio = punctuation_count / max(len(words), 1)
        capital_letters = len(re.findall(r'[A-Z]', text))
        capitalization_ratio = capital_letters / max(len(text.replace(" ", "")), 1)
        unique_word_ratio = len(set(words)) / len(words)
        return {
            'avg_sentence_length': avg_sentence_length,
            'vocab_richness': float(vocab_richness),
            'common_words_ratio': float(common_words_ratio),
            'sentence_length_variance': sentence_length_variance,
            'avg_word_length': avg_word_length,
            'word_length_variance': word_length_variance,
            'punctuation_ratio': float(punctuation_ratio),
            'capitalization_ratio': float(capitalization_ratio),
            'unique_word_ratio': float(unique_word_ratio)
        }

    def calculate_perplexity(self, text: str) -> Dict[str, float]:
        try:
            sentences = nltk.sent_tokenize(text)
            tokens = [w.lower() for w in nltk.word_tokenize(text) if any(c.isalpha() for c in w)]
            if len(tokens) < 20 or len(sentences) < 2:
                return {'score': 0.5, 'ppl': 0.0, 'burstiness': 0.0}
            
            if self._lm is None:
                try:
                    train_sents = [[w.lower() for w in sent] for sent in brown.sents()[:1000]]  # Limit corpus size
                    train_data, vocab = padded_everygram_pipeline(3, train_sents)
                    self._lm = KneserNeyInterpolated(3)
                    self._lm.fit(train_data, vocab)
                except Exception as e:
                    print(f"Error loading NLTK corpus: {e}")
                    return {'score': 0.5, 'ppl': 0.0, 'burstiness': 0.0}
            
            padded = list(pad_both_ends(tokens, n=3))
            grams = [tuple(padded[i-2:i+1]) for i in range(2, len(padded))]
            log_prob = 0.0
            for g in grams:
                prob = self._lm.score(g[-1], g[:-1])
                log_prob += math.log(max(prob, 1e-12))
            ppl = math.exp(-log_prob / max(len(grams), 1))

            sent_scores = []
            for s in sentences:
                toks = [w.lower() for w in nltk.word_tokenize(s) if any(c.isalpha() for c in w)]
                if len(toks) < 2:
                    continue
                sp = list(pad_both_ends(toks, n=3))
                gs = [tuple(sp[i-2:i+1]) for i in range(2, len(sp))]
                if not gs:
                    continue
                s_log = 0.0
                for g in gs:
                    s_log += math.log(max(self._lm.score(g[-1], g[:-1]), 1e-12))
                sent_scores.append(s_log / len(gs))
            burstiness = float(np.var(sent_scores)) if len(sent_scores) >= 2 else 0.0
            ppl_component = min(ppl / 600.0, 1.0)
            burst_component = max(0.0, min((burstiness + 5.0) / 10.0, 1.0))
            score = 0.6 * ppl_component + 0.4 * burst_component
            return {'score': float(max(0.0, min(1.0, score))), 'ppl': float(ppl), 'burstiness': float(burstiness)}
        except Exception as e:
            print(f"Error in perplexity calculation: {e}")
            return {'score': 0.5, 'ppl': 0.0, 'burstiness': 0.0}

    def get_ml_prediction(self, text: str) -> float:
        try:
            X = self.detector.vectorizer.transform([text])
            prob = self.detector.classifier.predict_proba(X)[0, 1]
            return float(prob)
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return 0.5

    def analyze_text(self, text: str, method: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        
        if method in ["Stylometric Analysis", "Combined Analysis"]:
            features = self.calculate_stylometric_features(text)
            stylometric_score = (
                (1 - features.get('common_words_ratio', 0)) * 0.3 +
                features.get('unique_word_ratio', 0) * 0.3 +
                (1 - features.get('punctuation_ratio', 0)) * 0.2 +
                features.get('sentence_length_variance', 0) / 100 * 0.2
            )
            stylometric_score = float(np.clip(stylometric_score, 0.0, 1.0))
            
            results['stylometric'] = {
                **features,
                'stylometric_score': stylometric_score
            }

        if method in ["Perplexity Analysis", "Combined Analysis"]:
            perplexity_details = self.calculate_perplexity(text)
            results['perplexity'] = {
                'perplexity': perplexity_details.get('score', 0.5),
                'perplexity_details': perplexity_details
            }

        if method in ["ML Classification", "Combined Analysis"]:
            ml_score = self.get_ml_prediction(text)
            results['ml'] = {
                'ml_score': ml_score
            }

        if method == "Combined Analysis":
            weights = { 'stylometric': 0.25, 'perplexity': 0.35, 'ml': 0.40 }
            stylometric_score = results.get('stylometric', {}).get('stylometric_score', 0.5)
            perplexity_score = results.get('perplexity', {}).get('perplexity', 0.5)
            ml_score = results.get('ml', {}).get('ml_score', 0.5)
            
            combined_score = (
                stylometric_score * weights['stylometric'] +
                perplexity_score * weights['perplexity'] +
                (1 - ml_score) * weights['ml']
            )
            results['combined'] = {
                'combined_score': float(np.clip(combined_score, 0.0, 1.0))
            }
            
        return results


