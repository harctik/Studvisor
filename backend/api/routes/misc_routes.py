"""Calendar, Documents, Bunk Alerts, Anon Chat — lightweight routes."""
from collections import defaultdict

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import AcademicCalendar, AnonPost, AnonReaction, Attendance, Subject
from backend.core.security import get_current_student, get_current_user

# Calendar
calendar_router = APIRouter(prefix="/calendar", tags=["Calendar"])
@calendar_router.get("/")
def get_calendar(db: Session = Depends(get_db)):
    days = db.query(AcademicCalendar).order_by(AcademicCalendar.date).all()
    return {"calendar": [{"date": d.date, "day": d.day_name, "working": d.is_working_day, "holiday": d.holiday_name, "note": d.note} for d in days]}

# Documents
documents_router = APIRouter(prefix="/documents", tags=["Documents"])
@documents_router.get("/")
def list_docs(student=Depends(get_current_student)):
    return {"documents": [{"type": "Bonafide Certificate", "url": f"/reports/bonafide/{student.id}"}, {"type": "Marksheet", "url": f"/reports/marksheet/{student.id}"}, {"type": "Attendance Certificate", "url": f"/reports/attendance-cert/{student.id}"}]}

# Bunk Alerts
bunk_alerts_router = APIRouter(prefix="/bunk-alerts", tags=["Bunk Alerts"])
@bunk_alerts_router.get("/")
def bunk_alerts(student=Depends(get_current_student), db: Session = Depends(get_db)):
    records = db.query(Attendance).filter(Attendance.student_id == student.id).all()
    data = defaultdict(lambda: {"t": 0, "p": 0})
    for r in records:
        data[r.subject_id]["t"] += 1
        if r.status == "P": data[r.subject_id]["p"] += 1
    alerts = []
    for sid, d in data.items():
        pct = round(d["p"]/d["t"]*100, 1) if d["t"] > 0 else 100
        if pct < 75:
            subj = db.query(Subject).filter(Subject.id == sid).first()
            alerts.append({"subject": subj.name if subj else "?", "pct": pct, "level": "critical" if pct < 65 else "warning"})
    return {"alerts": alerts}

# Anonymous Chat / Campus Wall
anon_chat_router = APIRouter(prefix="/anon", tags=["Anonymous"])

class AnonPostCreate(BaseModel):
    content: str
    category: str = "general"

@anon_chat_router.get("/posts")
def list_posts(_=Depends(get_current_user), db: Session = Depends(get_db)):
    posts = db.query(AnonPost).filter(AnonPost.moderated.is_(False)).order_by(AnonPost.created_at.desc()).limit(50).all()
    result = []
    for p in posts:
        reactions = db.query(AnonReaction).filter(AnonReaction.post_id == p.id).all()
        reaction_counts = defaultdict(int)
        for r in reactions: reaction_counts[r.reaction_type] += 1
        result.append({"id": p.id, "content": p.content, "category": p.category, "reactions": dict(reaction_counts), "date": str(p.created_at)})
    return {"posts": result}

@anon_chat_router.post("/posts")
def create_post(data: AnonPostCreate, student=Depends(get_current_student), db: Session = Depends(get_db)):

    from backend.services.anonymity_service import compute_anon_id
    from backend.services.sentiment_service import sentiment_service

    # Analyze sentiment
    analysis = sentiment_service.analyze(data.content)

    session_hash = compute_anon_id(student.id)
    post = AnonPost(
        session_hash=session_hash,
        content=data.content,
        category=data.category,
        moderated=analysis["needs_moderation"],
    )
    db.add(post)
    db.commit()

    return {
        "message": "Posted anonymously",
        "moderated": post.moderated,
        "warning": "Post pending moderation due to community guidelines" if post.moderated else None
    }
