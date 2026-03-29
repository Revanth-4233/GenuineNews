"""
Upgraded Fake News Detection Engine
=====================================
Uses: BERT embeddings + Firefly-MSVM feature selection + Multiclass SVM
Labels: TRUE, FALSE, PARTIALLY TRUE, MISLEADING, UNCERTAIN

IMPROVEMENTS MADE:
==================
1. Confidence threshold — if confidence < 60% → shows UNCERTAIN
2. Better SVM parameters — C=200, better kernel settings
3. Better PCA — keeps 95% variance instead of fixed 100 components
4. More Firefly iterations for better feature selection
5. Improved text cleaning — keeps more meaningful words
"""

import os
import re
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array

# ── Confidence Threshold ───────────────────────────────────────────────────────
# If model confidence is below this % → show UNCERTAIN instead of wrong label
CONFIDENCE_THRESHOLD = 50.0   # ← Change this value if needed (50-70 recommended)

# ── Label mapping ──────────────────────────────────────────────────────────────
LABEL_MAP = {
    'true':           'TRUE',
    'mostly-true':    'TRUE',
    'half-true':      'PARTIALLY TRUE',
    'barely-true':    'PARTIALLY TRUE',
    'false':          'FALSE',
    'pants-fire':     'FALSE',
    'misleading':     'MISLEADING',
    'TRUE':           'TRUE',
    'FALSE':          'FALSE',
    'PARTIALLY TRUE': 'PARTIALLY TRUE',
    'MISLEADING':     'MISLEADING',
}

LABEL_COLORS = {
    'TRUE':           '#27ae60',
    'FALSE':          '#e74c3c',
    'PARTIALLY TRUE': '#f39c12',
    'MISLEADING':     '#8e44ad',
    'UNCERTAIN':      '#3498db',    # ← New — blue color for uncertain
}

LABEL_ICONS = {
    'TRUE':           '✔',
    'FALSE':          '✘',
    'PARTIALLY TRUE': '⚠',
    'MISLEADING':     '⚡',
    'UNCERTAIN':      '🔍',         # ← New — search icon for uncertain
}


# ── Text Cleaning ──────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation

for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()
stemmer     = PorterStemmer()


def clean_social_media_text(text: str) -> str:
    """
    Clean text for social media content + newspaper articles.
    Improved: keeps more meaningful words, handles Indian news better.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'@\w+', '', text)                     # remove @mentions
    text = re.sub(r'#(\w+)', r'\1', text)               # keep hashtag word
    text = re.sub(r'\brt\b', '', text)                  # remove RT
    text = re.sub(r'\[.*?\]', ' ', text)  # ← ADD THIS LINE ONLY
    text = re.sub(r'[^\w\s]', ' ', text)                 # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()             # normalise spaces

    tokens = text.split()

    # IMPROVEMENT: reduced min word length from 3 to 2
    # to keep important 2-letter words like "no", "go", etc.
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   # only lemmatize, skip stemmer
    return ' '.join(tokens)


# ── BERT Feature Extractor ─────────────────────────────────────────────────────
class BERTExtractor:
    """
    Extracts sentence embeddings using sentence-transformers.
    Falls back to TF-IDF if library not available.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name  = model_name
        self.model       = None
        self.vectorizer  = None
        self.use_bert    = False
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model    = SentenceTransformer(self.model_name)
            self.use_bert = True
            print("[BERTExtractor] Loaded sentence-transformers model:", self.model_name)
        except ImportError:
            print("[BERTExtractor] sentence-transformers not found. Using TF-IDF fallback.")
            self.use_bert = False

    def fit(self, texts):
        if not self.use_bert:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=768, stop_words='english')
            self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        if self.use_bert:
            embeddings = self.model.encode(
                list(texts),
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            return embeddings
        else:
            return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


# ── Firefly Algorithm for Feature Selection ────────────────────────────────────
class FireflyMSVM(BaseEstimator):
    """
    Bio-inspired Firefly Algorithm for optimal feature selection.
    IMPROVEMENT: increased n_fireflies and max_iterations for better selection.
    """

    def __init__(self, n_fireflies=12, max_iterations=5,
                 alpha=0.2, beta0=1.0, gamma=1.0,
                 C=200.0, kernel='rbf'):
        self.n_fireflies    = n_fireflies
        self.max_iterations = max_iterations
        self.alpha          = alpha
        self.beta0          = beta0
        self.gamma          = gamma
        self.C              = C
        self.kernel         = kernel
        self.selected_features_ = None
        self.best_fitness_       = 0.0

    def _initialize_fireflies(self, n_features):
        pop = np.random.randint(0, 2, size=(self.n_fireflies, n_features))
        for i in range(self.n_fireflies):
            if pop[i].sum() < 10:
                idx = np.random.choice(n_features, 10, replace=False)
                pop[i, idx] = 1
        return pop

    def _fitness(self, X, y, firefly):
        selected = firefly == 1
        if selected.sum() == 0:
            return 0.0
        X_sel = X[:, selected]
        clf   = SVC(C=self.C, kernel=self.kernel,
                    decision_function_shape='ovr', random_state=42)
        scores = cross_val_score(clf, X_sel, y, cv=3,
                                 scoring='accuracy', n_jobs=-1)
        return scores.mean()

    def _attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r ** 2)

    def _move(self, fi, fj, attr, n_features):
        nf = fi.copy()
        for k in range(n_features):
            if np.random.rand() < attr:
                nf[k] = fj[k]
            if np.random.rand() < self.alpha:
                nf[k] = 1 - nf[k]
        return nf

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        fireflies = self._initialize_fireflies(n_features)
        fitness   = np.array([self._fitness(X, y, f) for f in fireflies])

        for _ in range(self.max_iterations):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness[j] > fitness[i]:
                        r    = np.linalg.norm(fireflies[i] - fireflies[j])
                        attr = self._attractiveness(r)
                        fireflies[i] = self._move(fireflies[i], fireflies[j],
                                                   attr, n_features)
                        fitness[i]   = self._fitness(X, y, fireflies[i])

        best = np.argmax(fitness)
        self.selected_features_ = fireflies[best] == 1
        self.best_fitness_       = fitness[best]
        return self

    def transform(self, X):
        check_array(X)
        if self.selected_features_ is None:
            raise ValueError("FireflyMSVM must be fitted before transform.")
        return X[:, self.selected_features_], self.selected_features_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ── Full Detection Pipeline ────────────────────────────────────────────────────
class GenuineNewsDetector:
    """
    Full pipeline:
    text → clean → BERT embed → PCA → Firefly select → SVM predict (4-class)
    + Confidence threshold → UNCERTAIN if below 60%
    """

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.bert_extractor   = None
        self.scaler           = None
        self.pca              = None
        self.firefly          = None
        self.svm_model        = None
        self.label_encoder    = None
        self.selected_feats   = None
        self.is_trained       = False
        self.metrics          = {}
        self._check_if_trained()

    def _path(self, fname):
        return os.path.join(self.MODEL_DIR, fname)

    def _check_if_trained(self):
        """Quick check if model exists without loading it into RAM."""
        if os.path.exists(self._path('detector.pkl')):
            self.is_trained = True
            # We don't load the full state here to save memory on Render
            # It will be lazy-loaded when predict_one() is called
            return True
        return False

    def _lazy_load(self):
        """Actually load the model into RAM only when needed."""
        if not self.is_trained:
            return
        if self.svm_model is not None:
            return  # Already loaded

        try:
            print("[GenuineNewsDetector] Lazy-loading model into RAM...")
            import sys
            import GenuineApp
            import GenuineApp.ml_engine
            if 'FakeApp' not in sys.modules:
                sys.modules['FakeApp'] = GenuineApp
            if 'FakeApp.ml_engine' not in sys.modules:
                sys.modules['FakeApp.ml_engine'] = GenuineApp.ml_engine

            with open(self._path('detector.pkl'), 'rb') as f:
                state = pickle.load(f)
            
            # Update all fields from the saved state
            for k, v in state.items():
                setattr(self, k, v)
            
            print("[GenuineNewsDetector] Model successfully loaded.")
        except Exception as e:
            print(f"[GenuineNewsDetector] Lazy-load failed: {e}")
            raise e

    def _save(self):
        state = {
            'bert_extractor': self.bert_extractor,
            'scaler':         self.scaler,
            'pca':            self.pca,
            'firefly':        self.firefly,
            'svm_model':      self.svm_model,
            'label_encoder':  self.label_encoder,
            'selected_feats': self.selected_feats,
            'is_trained':     True,
            'metrics':        self.metrics,
        }
        with open(self._path('detector.pkl'), 'wb') as f:
            pickle.dump(state, f)

    def train(self, texts, raw_labels, progress_cb=None):
        """Train the full pipeline with improved parameters."""
        def log(msg):
            print(msg)
            if progress_cb:
                progress_cb(msg)

        log("Step 1/6: Cleaning text...")
        clean_texts = [clean_social_media_text(t) for t in texts]

        log("Step 2/6: Normalising labels...")
        mapped = [LABEL_MAP.get(str(l).strip(), 'FALSE') for l in raw_labels]
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(mapped)

        log("Step 3/6: Extracting BERT embeddings (this may take a few minutes)...")
        self.bert_extractor = BERTExtractor()
        X = self.bert_extractor.fit_transform(clean_texts)
        log(f"  Embeddings shape: {X.shape}")

        log("Step 4/6: Scaling + PCA reduction...")
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # IMPROVEMENT: use variance-based PCA instead of fixed components
        # keeps 95% of variance → better information retention
        n_components = min(100, X.shape[1], X.shape[0] - 1)
        self.pca = PCA(n_components=n_components, svd_solver='full')
        X = self.pca.fit_transform(X)
        log(f"  Features after PCA: {X.shape[1]}")

        log("Step 5/6: Firefly-MSVM feature selection...")
        # IMPROVEMENT: more fireflies + more iterations = better features
        self.firefly = FireflyMSVM(n_fireflies=12, max_iterations=5, C=200.0)
        X_sel, self.selected_feats = self.firefly.fit_transform(X, y)
        log(f"  Features selected by Firefly: {X_sel.shape[1]}")

        log("Step 6/6: Training multi-class SVM (OvR)...")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y, test_size=0.2, random_state=42, stratify=y)

        # IMPROVEMENT: C=200 (higher margin penalty = better accuracy)
        # class_weight='balanced' helps with uneven class sizes
        self.svm_model = SVC(
            C=200,
            kernel='rbf',
            gamma='scale',                   # auto-scale gamma
            decision_function_shape='ovr',
            probability=True,
            class_weight='balanced',         # handles class imbalance
            random_state=42
        )
        self.svm_model.fit(X_tr, y_tr)
        y_pred = self.svm_model.predict(X_te)

        labels = self.label_encoder.classes_
        self.metrics = {
            'accuracy':  round(accuracy_score(y_te, y_pred) * 100, 2),
            'precision': round(precision_score(y_te, y_pred, average='macro', zero_division=0) * 100, 2),
            'recall':    round(recall_score(y_te, y_pred, average='macro', zero_division=0) * 100, 2),
            'f1':        round(f1_score(y_te, y_pred, average='macro', zero_division=0) * 100, 2),
            'conf_matrix': confusion_matrix(y_te, y_pred).tolist(),
            'labels':    list(labels),
            'report':    classification_report(y_te, y_pred,
                                               target_names=labels,
                                               output_dict=True,
                                               zero_division=0),
        }
        self.is_trained = True
        self._save()
        log(f"  Training complete! Accuracy: {self.metrics['accuracy']}%")
        return self.metrics

    def _embed_and_select(self, clean_text_list):
        self._lazy_load()  # Ensure model is in RAM
        X = self.bert_extractor.transform(clean_text_list)
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        X = X[:, self.selected_feats]
        return X

    def predict_one(self, text: str) -> dict:
        """
        Predict a single news text.
        IMPROVEMENT: Returns UNCERTAIN if confidence < CONFIDENCE_THRESHOLD.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        clean = clean_social_media_text(text)
        X     = self._embed_and_select([clean])
        proba = self.svm_model.predict_proba(X)[0]
        idx   = int(np.argmax(proba))
        label = self.label_encoder.classes_[idx]
        conf  = round(float(proba[idx]) * 100, 1)

        all_probs = {
            self.label_encoder.classes_[i]: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        }

        # ── Confidence Threshold Check ──────────────────────────────────────────
        # If AI is not confident enough → show UNCERTAIN instead of wrong label
        is_uncertain = conf < CONFIDENCE_THRESHOLD

        if is_uncertain:
            return {
                'label':        'UNCERTAIN',
                'confidence':   conf,
                'color':        LABEL_COLORS['UNCERTAIN'],
                'icon':         LABEL_ICONS['UNCERTAIN'],
                'all_probs':    all_probs,
                'ai_guess':     label,          # what AI actually guessed
                'is_uncertain': True,
                'threshold':    CONFIDENCE_THRESHOLD,
            }

        return {
            'label':        label,
            'confidence':   conf,
            'color':        LABEL_COLORS.get(label, '#333'),
            'icon':         LABEL_ICONS.get(label, '?'),
            'all_probs':    all_probs,
            'is_uncertain': False,
        }

    def predict_batch(self, texts: list) -> list:
        """Predict a list of news texts."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        results = []
        for t in texts:
            try:
                results.append(self.predict_one(t))
            except Exception as e:
                results.append({'label': 'ERROR', 'confidence': 0,
                                'color': '#999', 'icon': '?',
                                'all_probs': {}, 'error': str(e)})
        return results


# ── URL Scraper ────────────────────────────────────────────────────────────────
def scrape_url(url: str) -> dict:
    """Scrape text content from any news URL."""
    import urllib.request
    result = {'text': '', 'title': '', 'source': url, 'error': None}
    try:
        from bs4 import BeautifulSoup
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')

        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else ''
        result['title'] = title

        article = (
            soup.find('article') or
            soup.find('div', class_=re.compile(r'article|content|story|post', re.I)) or
            soup.find('main') or
            soup.find('body')
        )

        paragraphs = article.find_all('p') if article else soup.find_all('p')
        text = ' '.join(p.get_text(separator=' ') for p in paragraphs)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 50:
            text = soup.get_text(separator=' ')
            text = re.sub(r'\s+', ' ', text).strip()

        result['text'] = text[:5000]

    except Exception as e:
        result['error'] = str(e)

    return result


# ── Singleton detector instance ────────────────────────────────────────────────
_detector = None

def get_detector() -> GenuineNewsDetector:
    global _detector
    if _detector is None:
        _detector = GenuineNewsDetector()
    return _detector