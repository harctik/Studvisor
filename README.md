<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude_3-Anthropic-D97706?style=for-the-badge&logo=anthropic&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

# 🎓 STUDVISOR v3.0

**Next-Generation AI-Powered College ERP & Predictive Student Engagement Platform**

[Architecture](#-advanced-backend-architecture) • [Features](#-core-capabilities) • [Project Structure](#-project-structure) • [Deployment](#-deployment) • [CI/CD](#-cicd)

</div>

---

## 🏛️ Advanced Backend Architecture

The backend is built with **FastAPI** and **PostgreSQL**, utilizing a modular service-oriented architecture designed for multi-tenancy and high security.

### 🛡️ Security & Integrity Layer
- **Three-Layer RBAC**: JWT Claim verification → Route-level Guards → Database Scoping.
- **Audit Middleware**: Every state-changing request is logged with actor metadata, IP, and resource IDs.
- **Attendance Hardening**: 24-hour amendment window with an HOD-governed approval workflow for historical corrections.
- **Rotating HMAC Anonymity**: Daily-rotating salts for anonymous campus wall posts to prevent deanonymization.

### 🧠 AI Engine (Anthropic Claude-3 / Ollama)
- **Predictive Intervention**: Nightly batch jobs compute Dropout Risk, Academic Failure Risk, and Disengagement scores.
- **Context-Aware Chat**: Streaming AI assistant with real-time access to student performance data and institutional knowledge.
- **Plagiarism Detection**: High-fidelity semantic comparison of student assignments.
- **Auto-Question Generator**: Generates balanced exam papers based on syllabus coverage and Bloom's Taxonomy.
- **Sentiment Routing**: AI-powered grievance escalation for critical student complaints.

### 📅 Smart Academic Services
- **Conflict-Aware Timetable**: Automated scheduling engine that prevents room double-bookings and faculty overlaps.
- **Merit System**: Gamified engagement tracking with badges, XP, and leaderboards.
- **Placement Matcher**: AI-driven fitting of student profiles against open job drives.
- **GPA Forecaster**: Interactive GPA projections based on hypothetical targets.

---

## 📁 Project Structure

```text
Studvisor/
├── backend/
│   ├── api/
│   │   ├── routes/          # RBAC-guarded endpoint controllers
│   │   │   ├── admin.py     # Institutional management & analytics
│   │   │   ├── faculty.py   # Grading, attendance, & class health
│   │   │   ├── student.py   # Profiles, mood check-ins, recovery
│   │   │   ├── ai_engine.py # LLM orchestration & streaming
│   │   │   └── reports.py   # High-fidelity HTML/PDF generators
│   │   └── deps.py          # Dependency injection & auth
│   ├── core/
│   │   ├── security.py      # JWT, RBAC logic, & scope helpers
│   │   ├── middleware.py    # Audit logging & CORS
│   │   └── ai_context.py    # RAG system for LLM context
│   ├── services/            # Business logic & AI algorithms
│   │   ├── predictive.py    # Risk scoring algorithms
│   │   ├── merit.py         # Gamification engine
│   │   └── timetable.py     # Conflict resolution engine
│   └── app/
│       ├── models.py        # 45+ SQLAlchemy models
│       └── schemas.py       # Pydantic validation layer
├── frontend/                # Modern React / Vite / Tailwind UI
├── tests/                   # Automated unit & integration tests
├── alembic/                 # Database migrations
├── docker-compose.yml       # Containerized multi-service deployment
└── Dockerfile
```

---

## ⚡ Deployment

### Prerequisites
- Python 3.10+
- PostgreSQL
- Anthropic API Key (optional, falls back to Mock AI)

### Quick Start

```bash
# 1. Clone & Setup
git clone https://github.com/harctik/Studvisor.git
cd Studvisor
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure Environment
cp .env.example .env

# 3. Initialize Database
python -m backend.seed_all

# 4. Run Development Server
uvicorn backend.main:app --reload
```

### Docker Compose

```bash
docker-compose up --build
```

---

## 🧪 Testing & CI

```bash
pytest tests/ -v
```

All pushes and pull requests trigger automated linting with **Ruff** and tests across PostgreSQL and SQLite via GitHub Actions.

---

## 📄 License

This project is licensed under the MIT License.
