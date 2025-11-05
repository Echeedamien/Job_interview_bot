from flask import Flask, render_template, request, redirect, session, jsonify, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os, random, subprocess, json, re
from datetime import datetime
import spacy
import whisper
import pdfplumber
import torch
import docx2txt
from PyPDF2 import PdfReader
from docx import Document
import tempfile
import speech_recognition as sr
from pydub import AudioSegment

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from whisper import load_model

# ============================================================
# 🔧 APP CONFIGURATION
# ============================================================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

nlp = spacy.load("en_core_web_sm")
print("🔍 Initializing Whisper model…")

try:
    # Try to load from local cache folder (fully offline)
    model_path = os.path.expanduser(r"C:\Users\HP\.cache\whisper\small.pt")

    if os.path.exists(model_path):
        print("📁 Found local model file — loading directly.")
        whisper_model = whisper.load_model("tiny")
        print("✅ Whisper model loaded successfully.")
    else:
        print("⚠️ Model not found locally — downloading from official source.")
        whisper_model = whisper.load_model("small")
        print("✅ Whisper model downloaded and loaded successfully.")

except Exception as e:
    print(f"❌ Whisper model loading failed: {e}")
    whisper_model = None

# ============================================================
# 🧩 MODELS
# ============================================================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    role = db.Column(db.String(100))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    score = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class VoiceAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    history_id = db.Column(db.Integer, db.ForeignKey("history.id"), nullable=True)
    duration = db.Column(db.Float)
    words = db.Column(db.Integer)
    wpm = db.Column(db.Float)
    filler_count = db.Column(db.Integer)
    filler_rate = db.Column(db.Float)
    pauses = db.Column(db.Integer)
    avg_pause_sec = db.Column(db.Float)
    confidence = db.Column(db.Float)
    raw_metrics = db.Column(db.Text)
    star_score = db.Column(db.Float, default=0.0)
    star_breakdown = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============================================================
# 🧠 STAR Evaluation Helpers
# ============================================================
STAR_KEYWORDS = {
    "S": ["situation", "context", "when", "during"],
    "T": ["task", "goal", "responsibility", "objective"],
    "A": ["action", "led", "implemented", "performed", "took"],
    "R": ["result", "outcome", "impact", "improved", "achieved"]
}

def evaluate_star_method(text):
    doc = nlp(text.lower())
    found = {"S": False, "T": False, "A": False, "R": False}
    for tag, kws in STAR_KEYWORDS.items():
        for kw in kws:
            if kw in text.lower():
                found[tag] = True
    if not found["A"]:
        if any(tok.pos_ == "VERB" for tok in doc):
            found["A"] = True
    score = sum(found.values()) / 4.0
    return found, round(score, 2)

def choose_next_question(current_question, transcript, resume_skills=None, role=None):
    found, _ = evaluate_star_method(transcript or "")
    missing = [k for k, v in found.items() if not v]
    if missing:
        tag = missing[0]
        prompts = {
            "S": "Can you describe the situation or context of that example?",
            "T": "What was your task or goal in that scenario?",
            "A": "What specific actions did you take?",
            "R": "What was the outcome or measurable result?"
        }
        return prompts[tag]
    if resume_skills:
        return f"Tell me how you applied {resume_skills[0]} in a real-world project."
    if role:
        return f"What’s one improvement you’d make in a {role} role?"
    return "Interesting! Could you expand on that?"

# ============================================================
# 🧾 Resume Text Extraction
# ============================================================
def extract_text_from_resume(file):
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file.filename.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def extract_skills(text):
    tokens = [t.text.lower() for t in nlp(text) if t.is_alpha]
    common_skills = ["python", "excel", "communication", "leadership", "ai", "data", "customer service"]
    return [s for s in common_skills if s in tokens]

# ============================================================
# 🎙️ Audio Conversion & Transcription
# ============================================================
def convert_webm_to_wav(input_path):
    output_path = input_path.replace(".webm", ".wav")
    subprocess.run(["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-y", output_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def transcribe_audio(file_path):
    wav_path = convert_webm_to_wav(file_path)
    result = whisper_model.transcribe(wav_path, language="en")
    return result["text"]

# ============================================================
# 📋 ROUTES
# ============================================================
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    """Main page — automatically shows a question and allows voice/text answer"""
    roles = ["Software Engineer", "Customer Support", "Data Analyst", "Project Manager", "Designer"]

    # Get role from form or default
    selected_role = request.form.get("role") or session.get("role", "Software Engineer")
    session["role"] = selected_role

    # Load or generate a question
    if "question" not in session or request.method == "POST" and "next_question" in request.form:
        role_questions = {
            "Software Engineer": [
                "Explain the difference between REST and GraphQL.",
                "Describe a challenging bug you fixed recently.",
                "What’s your experience with version control?"
            ],
            "Customer Support": [
                "How do you handle an angry customer?",
                "Describe how you resolved a difficult support ticket.",
                "What does good customer service mean to you?"
            ],
            "Data Analyst": [
                "How do you clean and prepare messy data?",
                "What tools do you use for visualization?",
                "Explain how you handle missing data."
            ],
            "Project Manager": [
                "How do you manage conflicting priorities?",
                "Describe your leadership style.",
                "Tell me about a project you delivered under pressure."
            ],
            "Designer": [
                "What’s your design process?",
                "How do you handle client feedback?",
                "Tell me about your favorite project so far."
            ]
        }

        question = random.choice(role_questions[selected_role])
        session["question"] = question
    else:
        question = session["question"]

    feedback, score = None, None

    # Handle answer submission
    if request.method == "POST" and "submit_answer" in request.form:
        answer = request.form.get("answer", "").strip()

        if answer:
            # Save to history
            h = History(
                user_id=current_user.id,
                role=selected_role,
                question=question,
                answer=answer,
                score=random.randint(1, 3)
            )
            db.session.add(h)
            db.session.commit()

            feedback = "✅ Good effort! Keep improving your clarity and examples."
            score = h.score

    return render_template("index.html",
                           roles=roles,
                           selected_role=selected_role,
                           question=question,
                           feedback=feedback,
                           score=score)

# ---------- User Authentication ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect("/")
        else:
            flash("Invalid username or password.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# ---------- Resume Upload ----------
@app.route("/analyze_resume", methods=["GET", "POST"])
@login_required
def analyze_resume():
    import re
    import docx2txt
    from PyPDF2 import PdfReader

    if request.method == "GET":
        return render_template("resume_upload.html")

    file = request.files.get("resume")
    if not file:
        return render_template("resume_upload.html", feedback="⚠️ Please upload a file first!")

    filename = file.filename.lower()
    resume_text = ""

    # ✅ Extract text from supported file types
    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            resume_text += page.extract_text() or ""
    elif filename.endswith(".docx"):
        resume_text = docx2txt.process(file)
    elif filename.endswith(".txt"):
        resume_text = file.read().decode("utf-8", errors="ignore")
    else:
        return render_template("resume_upload.html", feedback="❌ Unsupported file type. Upload PDF, DOCX, or TXT.")

    # ✅ Simple skill detection — improved version
    common_skills = [
        "Python", "JavaScript", "HTML", "CSS", "AI", "Data Analysis", "Machine Learning", "Customer Service",
        "SQL", "Flask", "Django", "Networking", "Cybersecurity", "Excel", "Project Management",
        "Communication", "Leadership", "Problem Solving", "Teamwork", "Cloud Computing", "AWS", "Docker"
    ]
    detected_skills = [s for s in common_skills if re.search(r"\b" + re.escape(s) + r"\b", resume_text, re.IGNORECASE)]

    if not detected_skills:
        detected_skills = ["General Skills"]

    # ✅ Create multiple smart questions for each skill
    questions = []
    for skill in detected_skills:
        questions.extend([
            f"Tell me about your experience with {skill}.",
            f"What challenges have you faced when using {skill}?",
            f"Can you share a project where {skill} made a difference?",
            f"How would you improve your {skill} skills in your next role?",
            f"Describe a situation where you solved a problem using {skill}."
        ])

    # Save detected questions in session (to use later in mock interview)
    session["mock_questions"] = questions

    return render_template("resume_analysis.html", skills=detected_skills, questions=questions)

    # ✅ Clean & process text
    resume_text = re.sub(r'\s+', ' ', resume_text.strip())
    doc = nlp(resume_text.lower())

    # ------------------------------
    #  ADVANCED SKILL EXTRACTION
    # ------------------------------
    known_skills = [
        "python", "java", "c++", "html", "css", "javascript", "sql",
        "excel", "power bi", "tableau", "data analysis", "machine learning",
        "ai", "django", "flask", "react", "project management",
        "communication", "leadership", "customer service", "marketing",
        "sales", "research", "negotiation", "adaptability", "problem solving"
    ]

    found_skills = set()
    for skill in known_skills:
        if skill in resume_text.lower():
            found_skills.add(skill)

    # Add spaCy-detected skills/nouns
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "SKILL", "WORK_OF_ART"] and len(ent.text) < 25:
            found_skills.add(ent.text.lower())

    # Fallback heuristic
    if not found_skills:
        found_skills = {token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]}

    found_skills = list(found_skills)[:10]  # limit to top 10 skills

    # ------------------------------
    #  DYNAMIC QUESTION GENERATION
    # ------------------------------
    question_templates = [
        "Describe your experience using {skill}.",
        "What challenges have you faced working with {skill}?",
        "How do you apply {skill} in real-world scenarios?",
        "What’s your favorite thing about {skill}?",
        "Explain a project where you used {skill}.",
        "How do you stay updated on new trends in {skill}?",
        "Can you describe a time when {skill} helped you solve a major issue?",
        "What are common mistakes people make when using {skill}?",
        "How would you teach {skill} to a beginner?",
        "How does {skill} fit into your current career goals?"
    ]

    all_questions = []
    for skill in found_skills:
        for template in random.sample(question_templates, k=min(5, len(question_templates))):
            all_questions.append(template.format(skill=skill))
    random.shuffle(all_questions)
    all_questions = all_questions[:15]  # pick 15 best questions

    # ------------------------------
    #  SAVE SKILLS TO SESSION
    # ------------------------------
    session["analyzed_skills"] = found_skills
    session["generated_questions"] = all_questions
    session["current_question_index"] = 0

    return render_template("resume_interview.html", skills=found_skills, question=all_questions[0])

# ---------- Audio Upload & STAR Analysis ----------
@app.route("/mock/upload_audio", methods=["POST"])
@login_required
def upload_audio():
    file = request.files.get("audio")
    if not file:
        return jsonify({"ok": False, "error": "No audio file received"})

    file_path = os.path.join("instance", file.filename)
    os.makedirs("instance", exist_ok=True)
    file.save(file_path)

    try:
        transcript = transcribe_audio(file_path)
        star_found, star_score = evaluate_star_method(transcript)
        follow_up = choose_next_question("Tell me about yourself.", transcript)
        return jsonify({
            "ok": True,
            "transcript": transcript,
            "star_breakdown": star_found,
            "star_score": star_score,
            "follow_up": follow_up
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ---------- Dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    from sqlalchemy import func
    user_id = current_user.id

    # Fetch all history for this user
    history = History.query.filter_by(user_id=user_id).all()

    if not history:
        return render_template("dashboard.html", skills=[], avg_score=0, chart_data=[])

    # Group scores by skill
    skill_summary = (
        db.session.query(
            History.role,
            func.avg(History.score).label("avg_score"),
            func.count(History.id).label("count")
        )
        .filter(History.user_id == user_id)
        .group_by(History.role)
        .all()
    )

    # Format for display
    skills = [
        {"name": s.role, "avg": round(s.avg_score, 2), "count": s.count}
        for s in skill_summary
    ]

    avg_score = round(sum(s["avg"] for s in skills) / len(skills), 2)

    return render_template("dashboard.html", skills=skills, avg_score=avg_score)

@app.route("/mock", methods=["GET", "POST"])
@login_required
def mock_interview():
    """Handles bidirectional navigation (next/previous) for mock interview"""
    # Load questions (you can replace with resume-based questions)
    questions = session.get("mock_questions", [
        "Tell me about yourself.",
        "Why do you want to work here?",
        "Describe a challenge you overcame.",
        "Where do you see yourself in five years?",
        "What are your strengths and weaknesses?"
    ])

    # Initialize progress tracking
    index = session.get("current_question_index", 0)
    answers = session.get("answers", [""] * len(questions))

    # Handle button clicks
    if request.method == "POST":
        action = request.form.get("action")
        current_answer = request.form.get("answer", "").strip()

        # Save current answer
        if 0 <= index < len(questions):
            answers[index] = current_answer

        # Handle navigation
        if action == "next" and index < len(questions) - 1:
            index += 1
        elif action == "previous" and index > 0:
            index -= 1
        elif action == "finish":
            # Save all answers to DB
            for i, q in enumerate(questions):
                if answers[i].strip():
                    h = History(
                        user_id=current_user.id,
                        role="Mock Interview",
                        question=q,
                        answer=answers[i],
                        score=random.randint(1, 3)
                    )
                    db.session.add(h)
            db.session.commit()
            session.pop("answers", None)
            session.pop("current_question_index", None)
            flash("🎉 Mock Interview Completed!")
            return redirect("/dashboard")

        # Save back to session
        session["answers"] = answers
        session["current_question_index"] = index

    # Prepare data for rendering
    question = questions[index] if index < len(questions) else None
    previous_answer = answers[index] if 0 <= index < len(answers) else ""
    progress = f"{index + 1} of {len(questions)}"

    return render_template(
        "mock_session.html",
        question=question,
        answer=previous_answer,
        index=index + 1,
        total=len(questions),
        progress=progress
    )

# ============================================================
# 🎤 VOICE ANALYTICS DASHBOARD
# ============================================================

import json
from flask import render_template, flash

@app.route("/voice_analytics")
@login_required
def voice_analytics():
    """Display voice analytics safely without JSON serialization errors."""
    analyses = VoiceAnalysis.query.join(History).filter(History.user_id == current_user.id).all()

    if not analyses:
        flash("No voice analytics data found. Record some answers first.", "info")
        return render_template("voice_analytics.html", data=None)

    def safe_num(value, default=0.0):
        """Safely convert to float."""
        try:
            if value in [None, "undefined", "null", ""]:
                return default
            return float(value)
        except Exception:
            return default

    def safe_json(value):
        """Safely parse JSON or return an empty dict."""
        try:
            if not value or value in [None, "undefined", "null", ""]:
                return {}
            if isinstance(value, dict):
                return value
            return json.loads(value)
        except Exception:
            return {}

    details = []
    for a in analyses:
        details.append({
            "question": getattr(a.history, "question", "N/A"),
            "confidence": safe_num(getattr(a, "confidence", 0)),
            "filler_rate": safe_num(getattr(a, "filler_rate", 0)),
            "wpm": safe_num(getattr(a, "wpm", 0)),
            "metrics": safe_json(getattr(a, "raw_metrics", "{}")),
            "breakdown": safe_json(getattr(a, "star_breakdown", "{}"))
        })

    # Compute averages safely
    total = len(details)
    avg_confidence = round(sum(d["confidence"] for d in details) / total, 2) if total else 0
    avg_filler_rate = round(sum(d["filler_rate"] for d in details) / total, 2) if total else 0
    avg_wpm = round(sum(d["wpm"] for d in details) / total, 2) if total else 0

    data = {
        "count": total,
        "avg_confidence": avg_confidence,
        "avg_filler_rate": avg_filler_rate,
        "avg_wpm": avg_wpm,
        "details": details
    }

    # 💡 Never serialize to JSON before rendering — just pass raw Python objects.
    return render_template("voice_analytics.html", data=data)

@app.route("/voice_answer", methods=["POST"])
@login_required
def voice_answer():
    """Handle uploaded voice recordings, transcribe, and return text"""
    try:
        if "voice" not in request.files:
            return jsonify({"error": "No voice file received"}), 400

        file = request.files["voice"]
        if not file:
            return jsonify({"error": "Empty file"}), 400

        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", "temp_voice.webm")
        file.save(file_path)

        # ✅ Local Whisper transcription
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        text = result.get("text", "").strip()

        os.remove(file_path)

        if not text:
            return jsonify({"error": "No speech detected"}), 200

        return jsonify({"transcribed_text": text})
    except Exception as e:
        print("❌ Voice processing error:", e)
        return jsonify({"error": str(e)}), 500
# ============================================================

# Simple DB reset helper route (dev only) — protected
@app.route("/admin/reset_db")
@login_required
def reset_db():
    # only allow if username is 'admin' (or remove check as needed)
    if current_user.username != "admin":
        flash("Not authorized to reset DB", "danger")
        return redirect(url_for("dashboard"))
    # drop and recreate (development)
    db.drop_all()
    db.create_all()
    flash("Database reset — please register new accounts.", "info")
    return redirect(url_for("login"))

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username").strip()
        new_password = request.form.get("new_password").strip()
        user = User.query.filter_by(username=username).first()

        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("✅ Password reset successful. You can now login.", "success")
            return redirect("/login")
        else:
            flash("❌ Username not found.", "error")

    return render_template("forgot_password.html")

@app.route("/skill/<skill_name>")
@login_required
def view_skill(skill_name):
    """View all answers, scores, and progress for a specific skill"""
    from urllib.parse import unquote
    skill_name = unquote(skill_name)

    # Fetch Q&A history for that skill
    history = History.query.filter_by(user_id=current_user.id, role=skill_name).all()
    if not history:
        flash(f"No records found for {skill_name}. Try analyzing a resume first.")
        return redirect("/dashboard")

    # Compute simple stats
    total_score = sum(h.score for h in history)
    avg_score = round(total_score / len(history), 2)

    return render_template(
        "skill_view.html",
        skill_name=skill_name,
        history=history,
        avg_score=avg_score
    )

@app.route("/resume_mock_start", methods=["POST"])
@login_required
def resume_mock_start():
    """Start a mock interview for a skill based on resume analysis"""
    skill_name = request.form.get("skill_name")
    if not skill_name:
        flash("Skill name missing.")
        return redirect("/dashboard")

    # Fetch previous resume-generated questions
    questions = [
        h.question for h in History.query.filter_by(
            user_id=current_user.id, role=skill_name
        ).all()
    ]

    if not questions:
        flash("No resume-based questions found for this skill.")
        return redirect("/dashboard")

    # Reset session for the mock interview
    session["mock_questions"] = questions
    session["current_question_index"] = 0
    flash(f"Starting resume-based mock for {skill_name}!")
    return redirect("/mock")

# ============================================================
# 🚀 RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)