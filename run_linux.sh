#!/bin/bash
echo "============================================"
echo "  GenuineNews AI Detector - Upgraded Version"
echo "============================================"
echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Setting up database..."
python manage.py makemigrations
python manage.py migrate

echo ""
echo "Step 3: Starting web server..."
echo "Open your browser at: http://127.0.0.1:8000"
echo ""
python manage.py runserver
