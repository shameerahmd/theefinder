🔥 TheeFinder

AI-Based Detection & Classification of Industrial Fires and Persistent Thermal Sources

From Thermal Hotspots to Actionable Industrial Fire Intelligence

TheeFinder is an AI + GIS decision-support prototype designed to classify satellite-detected thermal anomalies and help distinguish potential industrial fires from persistent industrial thermal sources, forest fires, and agricultural/open burning.

📌 Problem Statement

Industrial facilities such as refineries, power plants, steel plants, petrochemical units, furnaces, boilers, kilns, and flare stacks can generate legitimate thermal signatures.

Satellite thermal-detection systems can identify hotspots, but a hotspot near an industrial area does not automatically mean an accidental fire.

TheeFinder adds a contextual intelligence layer that asks:

Is the hotspot industrial, forest, or agricultural/open burn?

If industrial, does it resemble a potential industrial fire or a persistent industrial heat source?

Is the thermal activity new or recurring?

What is the surrounding land cover?

How close is the detection to mapped industrial infrastructure?

💡 Proposed Solution

TheeFinder combines:

NASA FIRMS / VIIRS thermal hotspot data

Industrial GIS context from OpenStreetMap

ESA WorldCover land-cover information

Historical thermal behaviour

Two-stage Random Forest classification

PostGIS-based spatial storage and querying

Interactive GIS dashboard for live and historical analysis

The system is designed as a decision-support tool. It does not declare an industrial accident as confirmed; instead, it flags suspicious detections as Potential Industrial Fire for further verification.

⚙️ How It Works

flowchart LR
A[NASA FIRMS<br/>Thermal Hotspot] --> B[Feature Enrichment]

    C[OpenStreetMap<br/>Industrial Context] --> B
    D[ESA WorldCover<br/>Land Cover] --> B
    E[Historical FIRMS<br/>Thermal Behaviour] --> B

    B --> F[Stage A<br/>Random Forest]

    F --> G[Industrial]
    F --> H[Forest]
    F --> I[Agricultural / Open Burn]

    G --> J[Stage B<br/>Random Forest]

    J --> K[Potential Industrial Fire]
    J --> L[Persistent Industrial Thermal Source]

    H --> M[Final Classification]
    I --> M
    K --> M
    L --> M

    M --> N[(PostGIS)]
    N --> O[GIS Dashboard]

🧠 AI Classification Architecture

TheeFinder uses a two-stage hierarchical Random Forest model implemented with scikit-learn.

Stage A — Broad Source Classification

Classifies each thermal anomaly into:

INDUSTRIAL

FOREST

AGRICULTURAL_OPEN

Stage B — Industrial Event Classification

Executed only when Stage A predicts INDUSTRIAL.

It distinguishes between:

INDUSTRIAL_FIRE

PERSISTENT_INDUSTRIAL_SOURCE

The user-facing result is intentionally conservative:

POTENTIAL_INDUSTRIAL_FIRE

PERSISTENT_INDUSTRIAL_SOURCE

FOREST_FIRE

AGRICULTURAL_OPEN_BURN

UNCERTAIN_INDUSTRIAL_EVENT

🔬 Feature Fusion

The model combines several feature groups.

Thermal Features

Fire Radiative Power (FRP)

FIRMS confidence

Day / night information

Satellite information

Industrial Context

Distance to nearest mapped industrial feature

Number of industrial features within the search radius

Industrial proximity score

Land-Cover Context

Tree cover

Cropland

Built-up area

Shrubland

Grassland

Bare / sparse vegetation

Water / wetland classes

Historical Behaviour

Previous detections near the same location

Active thermal days

Persistence score

Historical FRP statistics

🌍 Data Sources

Source

Purpose

NASA FIRMS / VIIRS

Near-real-time thermal hotspot detections

OpenStreetMap / Overpass

Industrial facilities and GIS context

ESA WorldCover

Land-cover classification

Historical FIRMS detections

Persistence and thermal-history features

Verified incident samples

Ground-truth data for ML training

🗺️ GIS & Historical Intelligence

TheeFinder stores classified detections in PostgreSQL + PostGIS.

This enables:

Spatial bounding-box searches

Classification-based filtering

Historical detection retrieval

Geographic querying

Detection deduplication

Live vs historical comparison

🖥️ Dashboard Capabilities

The web dashboard supports:

Live Chennai thermal detections

Historical PostGIS detections

Interactive Leaflet map

Detection details

Stage A confidence

Stage B confidence

Final classification

FRP and acquisition information

Industrial proximity

Land-cover context

Persistence indicators

Classification filters

Contextual explanations

🧰 Technology Stack

Backend

Python

FastAPI

SQLAlchemy

GeoAlchemy2

Psycopg

Pandas

NumPy

Machine Learning

scikit-learn

Random Forest

Stratified Group K-Fold validation

Joblib model persistence

Geospatial

PostgreSQL

PostGIS

GeoPandas

Shapely

PyProj

OpenStreetMap / Overpass

ESA WorldCover

Frontend

Next.js

React

TypeScript

Tailwind CSS

Leaflet

Infrastructure

Docker

Docker Compose

Cloudflare Tunnel

Git / GitHub

📊 Prototype Dataset

The current verified prototype dataset contains 76 labelled samples:

Stage A Class

Samples

Industrial

27

Forest

24

Agricultural / Open Burn

25

Total

76

Industrial samples used for Stage B:

Stage B Class

Samples

Persistent Industrial Source

22

Verified Industrial Fire

5

Total Industrial Samples

27

⚠️ Important: Stage B currently contains only five verified industrial-fire positives. Its results should therefore be treated as prototype-level evidence, not production-grade industrial-fire accuracy.

📈 Model Validation

Stage A

Group-aware 5-fold validation:

Accuracy: ~89.5%

Balanced Accuracy: ~89.6%

Macro F1: ~89.5%

Stage B

Prototype industrial-event classifier:

Overall Accuracy: ~85.2%

The Stage B figure must not be interpreted as “85% industrial-fire detection accuracy” because the verified industrial-fire sample count is still small.

🏗️ Project Structure

theefinder/
├── backend/
│ ├── app/
│ │ ├── api/
│ │ ├── repositories/
│ │ └── services/
│ └── pyproject.toml
│
├── database/
│ └── init/
│
├── data/
│ └── training/
│
├── frontend/
│ └── src/
│ ├── app/
│ └── components/
│
├── ml/
│ ├── models/
│ └── training/
│
├── docker-compose.yml
├── start_theefinder.ps1
├── .gitignore
└── README.md

🚀 Running TheeFinder Locally

Prerequisites

Install:

Python 3.12+

Node.js

Docker Desktop

Git

uv

cloudflared (optional, for public demo access)

1. Clone the Repository

git clone https://github.com/shameerahmd/theefinder.git
cd theefinder

2. Configure Environment Variables

Create:

backend/.env

Use your local configuration based on .env.example.

Typical variables include:

FIRMS_MAP_KEY=your_firms_key_here
DATABASE_URL=your_postgresql_connection_string

Never commit backend/.env or real API/database credentials.

3. Start PostGIS

docker compose up -d

4. Start FastAPI

uv run --project backend fastapi dev backend/app/main.py

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

5. Start the Frontend

cd frontend
npm install
npm run build
npm run start -- --hostname 0.0.0.0

Dashboard:

http://localhost:3000

⚡ Windows One-Command Startup

The repository includes:

start_theefinder.ps1

Run:

powershell -ExecutionPolicy Bypass -File ".\start_theefinder.ps1"

The startup workflow is:

PostGIS
↓
FastAPI
↓
Next.js
↓
API Validation
↓
Cloudflare Tunnel

🌐 Public Demo Access

For temporary internet access:

cloudflared tunnel --url http://127.0.0.1:3000

Cloudflare will generate a temporary:

https://xxxxx.trycloudflare.com

Quick Tunnel URLs are temporary and may change after restart.

For production deployment, use a permanent hosted frontend/backend/database architecture with authentication, rate limiting, monitoring, and a named domain/tunnel.

✅ Prototype Status

The prototype currently demonstrates:

✅ NASA FIRMS integration

✅ Live VIIRS hotspot retrieval

✅ Industrial GIS enrichment

✅ ESA WorldCover enrichment

✅ Historical thermal persistence analysis

✅ Stage A ML classification

✅ Stage B industrial classification

✅ Confidence scores

✅ Context-based explanations

✅ PostgreSQL + PostGIS storage

✅ Historical GIS queries

✅ Interactive map dashboard

✅ Live / History modes

✅ Public demo access through Cloudflare Tunnel

⚠️ Limitations

Current prototype limitations include:

Small verified industrial-fire dataset

Dependence on satellite revisit / data availability

VIIRS spatial resolution limitations

Incomplete industrial mapping in OpenStreetMap

External GIS/API availability

No direct on-site sensor confirmation

No future fire forecasting in the current prototype

TheeFinder is currently a detection, classification and decision-support system, not a future fire-prediction system.

🔮 Future Scope

Planned improvements include:

Larger verified industrial-accident ground-truth database

Permanent industrial GIS dataset

Higher-resolution satellite imagery

Weather and wind integration

Temporal anomaly modelling

Automated alerting

Risk scoring

Human-in-the-loop verification

Nationwide configurable AOIs

Cloud deployment

Mobile notifications

Explainable-AI enhancements

🎯 Key Idea

NASA FIRMS tells us where thermal activity exists. TheeFinder helps estimate what that thermal anomaly may represent.

TheeFinder aims to move the workflow from:

DETECT
↓
UNDERSTAND
↓
CLASSIFY
↓
EXPLAIN
↓
STORE
↓
PRIORITISE

🛡️ Responsible Use

TheeFinder should not be used as the sole basis for declaring an industrial accident or initiating an emergency response.

A result such as Potential Industrial Fire indicates that a thermal anomaly deserves additional verification using appropriate sources such as:

Facility operators

Emergency services

Higher-resolution imagery

Ground sensors

Thermal cameras

Other verified incident information

📄 License

No open-source license has been selected yet.

Until a license is added, standard copyright restrictions apply to the repository.

👨‍💻 Project

TheeFinder
AI-Based Detection & Classification of Industrial Fires and Persistent Thermal Sources

Built as an AI + GIS prototype for industrial thermal intelligence.
