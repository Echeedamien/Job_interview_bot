# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    role = db.Column(db.String(100))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    score = db.Column(db.Integer)
    audio_path = db.Column(db.String(300))   # optional path to saved audio
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class VoiceAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    history_id = db.Column(db.Integer, db.ForeignKey('history.id'), nullable=False)
    duration = db.Column(db.Float)            # seconds
    words = db.Column(db.Integer)
    wpm = db.Column(db.Float)
    filler_count = db.Column(db.Integer)
    filler_rate = db.Column(db.Float)         # fillers per 100 words
    pauses = db.Column(db.Integer)            # number of pauses longer than threshold
    avg_pause_sec = db.Column(db.Float)       # average pause length in seconds
    confidence = db.Column(db.Float)          # heuristic 0-1
    raw_metrics = db.Column(db.Text)          # optional JSON dump of metrics
    # NEW fields for Phase 9:
    star_score = db.Column(db.Float, default=0.0)    # 0..1
    star_breakdown = db.Column(db.Text)              # JSON {S: bool, T: bool, A: bool, R: bool}
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SkillProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    skill_name = db.Column(db.String(100))
    total_answers = db.Column(db.Integer, default=0)
    average_score = db.Column(db.Float, default=0.0)
