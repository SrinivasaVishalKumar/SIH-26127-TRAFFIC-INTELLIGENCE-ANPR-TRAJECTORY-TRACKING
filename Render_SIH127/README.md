# Render_SIH127 — Traffic Intelligence Backend API

Flask-based REST backend service for SIH-26127 Traffic Intelligence, configured for deployment on Render.

## Features
- **Alerts API** (outes/alerts.py\): Real-time traffic violation and event alerts
- **Analytics API** (outes/analytics.py\): Aggregated vehicle counts, density, and flow metrics
- **Dashboard API** (outes/dashboard.py\): Data endpoints for frontend visualization
- **Tracking API** (outes/tracking.py\): Vehicle trajectory and ANPR record queries
- **Production Server**: Configured with Gunicorn and \Procfile\ for deployment on Render

## Setup & Running Locally
\\ash
python -m venv venv
venv\Scriptsctivate  # Windows
pip install -r requirements.txt
python app.py
\