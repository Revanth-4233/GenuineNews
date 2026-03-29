#!/usr/bin/env bash
# Render Build Script for GenuineNews

set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Download trained model from Google Drive
mkdir -p model
gdown "1oB9z6oEUfV7PCXxCQmZ4bHk3TQ7-lvLq" -O model/detector.pkl

# Collect static files
python manage.py collectstatic --no-input

# Run database migrations
python manage.py migrate --no-input
