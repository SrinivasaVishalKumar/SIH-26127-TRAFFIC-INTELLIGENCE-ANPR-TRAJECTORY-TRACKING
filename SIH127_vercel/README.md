# SIH127_vercel — Serverless Traffic Ingestion API

Serverless Python backend for SIH-26127 Traffic Intelligence, configured for deployment on Vercel.

## Features
- **Serverless API** (\pi/index.py\): Ingests real-time vehicle logs and camera telemetry
- **Vercel Configuration** (\ercel.json\): Routes API requests seamlessly
- **Database Integration**: Connects with cloud MongoDB to store plate logs and traffic counts

## Deployment
Deploy directly to Vercel via CLI or GitHub integration:
\\ash
vercel deploy
\