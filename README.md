# Genuine News Prediction Using Optimized MSVM

An advanced Fake News Detection platform that uses a hybrid approach: **Machine Learning** (Optimized Support Vector Machines) and **Real-Time Web Evidence Validation**.

## 🚀 Key Features
- **AI Prediction:** Uses an Optimized MSVM and TF-IDF vectorization to analyze news text and generate a detailed probability breakdown.
- **Real-Time Web Evidence:** Scrapes the internet using Google Custom Search, NewsAPI, and GDELT to cross-verify the AI's prediction against live news sources, automatically flagging if a story is corroborated by trusted sources or contradicted.
- **URL Scraping:** Users can input a news article URL. The system automatically extracts the article content using BeautifulSoup and analyzes it.
- **Batch Processing:** Allows users to upload a CSV file of news headlines/statements for bulk prediction processing.
- **Detailed Dashboards:** Users can view historical predictions and detailed model evaluation metrics (Accuracy, F1-Score, Confusion Matrix).
- **Optimized Memory Management:** The 90MB+ machine learning model is lazy-loaded into RAM only when predictions are actively requested, preventing memory crashes during deployment.

## 💻 Technology Stack
- **Backend:** Python, Django Web Framework
- **Machine Learning:** Scikit-Learn (SVM, TF-IDF), NLTK, Pandas, Numpy
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla custom styling)
- **Database:** SQLite3
- **External Integrations:** Google Search API, NewsAPI
- **Deployment:** Gunicorn, Whitenoise, Render Cloud

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd GenuineNews
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Start the Django Server:**
   ```bash
   python manage.py runserver
   ```
## 🌐 Deployment (Render)
This project is pre-configured for deployment on Render.
- Uses `build.sh` to install NLTK dependencies, collect static files, and run database migrations.
- Automatically downloads the pre-trained ML model (`detector.pkl`) directly from Google Drive via `gdown`, avoiding GitHub file size limits.
- Static files are served highly efficiently via `Whitenoise`.

## 🛠️ Troubleshooting: Fixing "Errors" in VS Code

If you see red squiggly lines in `views.py` or `ml_engine.py` (like "Import Django could not be resolved"), these are **not** code errors. They are VS Code environment warnings.

**To fix them:**
1. Open the command palette (`Ctrl + Shift + P`).
2. Type **"Python: Select Interpreter"**.
3. Select the Python interpreter inside your virtual environment (`.\venv\Scripts\python.exe`).
4. VS Code will now recognize all installed packages (Django, Pandas, Scikit-learn), and the red lines will disappear.

## 📊 Project Diagrams
Detailed system diagrams (Use Case, Class, Sequence, etc.) can be found in `Project_Diagrams.html`. 
- Open this file in your browser to view the diagrams.
- Print to PDF (`Ctrl + P`) to save them as a document.
