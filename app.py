from flask import Flask, render_template, request, session, redirect, url_for, send_file
import sqlite3
import joblib
import numpy as np
import io
import ast
import pandas as pd
import shap 
import re
import time
import os 
import ast



from flask import flash
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from flask_mail import Mail, Message
import random
import string


from database.db_operations import save_prediction 

from werkzeug.security import generate_password_hash, check_password_hash

# ======================================================
# APP CONFIG
# ======================================================
app = Flask(__name__)
app.config["SESSION_COOKIE_SECURE"] = True
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key")




# ================= EMAIL CONFIG =================

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)



# ======================================================
# UTILITIES
# ======================================================

def login_required():
    return "user_email" in session

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))


def get_float(form, field, default=0.0):
    try:
        return float(form.get(field, default))
    except:
        return default

def safe_float(val):
    try:
        return float(val)
    except:
        try:
            return float(val.decode(errors="ignore"))
        except:
            return 0.0

def is_strong_password(password):
    """
    Enforces strong password policy:
    - Min 8 chars
    - Uppercase
    - Lowercase
    - Digit
    - Special character
    """
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[^A-Za-z0-9]", password):
        return False
    return True

def compute_shap_percentages(explainer, X, feature_names):
    """
    SAFE SHAP handler for binary XGBoost models.
    Returns percentage contribution per feature.
    """
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values

    shap_vals = shap_vals[0]

    abs_vals = np.abs(shap_vals)
    total = abs_vals.sum() if abs_vals.sum() != 0 else 1.0

    return {
        feature_names[i]: float(round((abs_vals[i] / total) * 100, 2))
        for i in range(len(feature_names))
    }

def top_5_shap(shap_data: dict):
    """
    Keep top 5 contributors and combine rest as 'Others'
    """
    sorted_items = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_items[:5]
    others = sorted_items[5:]

    final_shap = dict(top_5)

    if others:
        final_shap["Others"] = round(sum(v for _, v in others), 2)

    return final_shap

def generate_shap_explanation(disease, shap_data):
    """
    Disease-specific human readable SHAP explanation
    """
    items = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)

    if not items:
        return "No significant contributing factors identified."

    if len(items) == 1:
        f, v = items[0]
        return f"{f.replace('_',' ').title()} ({v}%) is the main contributing factor."

    f1, v1 = items[0]
    f2, v2 = items[1]

    f1 = f1.replace("_", " ").title()
    f2 = f2.replace("_", " ").title()

    v1 = round(v1, 1)
    v2 = round(v2, 1)

    if disease == "diabetes":
        return (
            f"High {f1.lower()} ({v1}%) and {f2.lower()} ({v2}%) "
            "are the primary contributors to diabetes risk."
        )

    v1 = round(v1, 1)
    v2 = round(v2, 1)

    if disease == "heart":
        return (
            f"{f1} ({v1}%) and {f2.lower()} ({v2}%) "
            "are the major factors contributing to heart disease risk."
        )
    v1 = round(v1, 1)
    v2 = round(v2, 1)

    if disease == "kidney":
        return (
            f"{f1} ({v1}%) and {f2.lower()} ({v2}%) "
            "strongly influence kidney disease risk."
        )

    return (
        f"{f1} ({v1}%) and {f2} ({v2}%) "
        "are the key contributors to the prediction."
    )


# ======================================================
# ✅ HEALTH SUGGESTIONS LOGIC (NEW)
# ======================================================

def generate_health_suggestions(disease, risk, shap_data):

    if risk == "Low Risk":
        return {
            "level": "low",
            "title": "You are currently at low health risk",
            "message": "Maintain a healthy lifestyle to stay protected.",
            "items": [
                "Follow a balanced and nutritious diet",
                "Engage in regular physical activity",
                "Stay hydrated throughout the day",
                "Go for routine health checkups"
            ]
        }

    DIABETES_MAP = {
        "bmi": "Reduce calorie intake and increase fiber-rich foods.",
        "glucose": "Avoid sugary foods and refined carbohydrates.",
        "bp": "Limit salt intake and manage stress.",
        "age": "Maintain regular meal timing and monitoring."
    }

    HEART_MAP = {
        "chol": "Reduce saturated fats and fried foods.",
        "bp": "Follow a low-sodium diet.",
        "age": "Engage in light daily physical activity.",
        "thalach": "Maintain cardiovascular fitness through walking."
    }

    KIDNEY_MAP = {
        "renal_function": "Limit protein intake as advised.",
        "electrolyte_balance": "Avoid foods high in sodium and potassium.",
        "blood_pressure": "Control salt intake strictly.",
        "urine_sugar": "Monitor blood sugar regularly."
    }

    disease_map = (
        DIABETES_MAP if disease == "diabetes"
        else HEART_MAP if disease == "heart"
        else KIDNEY_MAP
    )

    suggestions = []
    for feature in shap_data.keys():
        if feature in disease_map:
            suggestions.append(disease_map[feature])

    if not suggestions:
        suggestions.append("Maintain a balanced diet and healthy lifestyle.")

    if risk == "High Risk":
        return {
            "level": "high",
            "title": "High Risk Detected",
            "message": "Please consult a medical professional immediately.",
            "items": suggestions
        }

    return {
        "level": "medium",
        "title": "Moderate Risk Detected",
        "message": "Lifestyle and dietary changes can help reduce future risk.",
        "items": suggestions
    }


# ======================================================
# ROUTES
# ======================================================

@app.route("/")
def home():
    return render_template("index.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if not is_strong_password(password):
            flash(
                "Password must be at least 8 characters and include uppercase, lowercase, number, and special character.",
                "error"
            )
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("database/health_data.db")
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM users WHERE email=?", (email,))
        if cursor.fetchone():
            conn.close()
            flash("User already exists. Please login.", "error")
            return redirect(url_for("login"))

        # ✅ STORE HASHED PASSWORD
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, hashed_password)
        )
        conn.commit()
        conn.close()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("database/health_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE email=?", (email,))
        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session["user_email"] = email
            return redirect(url_for("dashboard"))

        flash("Invalid login credentials", "error")
        return redirect(url_for("login"))

    return render_template("login.html")

# ---------------- PROFILE PAGE ----------------
@app.route("/profile")
def profile():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("profile.html", user=session["user_email"])


# ---------------- UPDATE EMAIL ----------------
@app.route("/update_email", methods=["POST"])
def update_email():
    if not login_required():
        return redirect(url_for("login"))

    new_email = request.form["new_email"].strip()
    password = request.form["password"]

    # Email format validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", new_email):
        flash("Invalid email format.", "error")
        return redirect(url_for("profile"))

    if new_email == session["user_email"]:
        flash("New email must be different from current email.", "error")
        return redirect(url_for("profile"))

    conn = sqlite3.connect("database/health_data.db")
    cursor = conn.cursor()

    # Verify password
    cursor.execute("SELECT password FROM users WHERE email=?", (session["user_email"],))
    row = cursor.fetchone()

    if not row or not check_password_hash(row[0], password):
        conn.close()
        flash("Incorrect password.", "error")
        return redirect(url_for("profile"))

    # Check duplicate email
    cursor.execute("SELECT 1 FROM users WHERE email=?", (new_email,))
    if cursor.fetchone():
        conn.close()
        flash("Email already in use.", "error")
        return redirect(url_for("profile"))

    # Update email in both tables
    cursor.execute("UPDATE users SET email=? WHERE email=?", (new_email, session["user_email"]))
    cursor.execute("UPDATE user_predictions SET user_id=? WHERE user_id=?", (new_email, session["user_email"]))

    conn.commit()
    conn.close()

    session.clear()
    flash("Email updated successfully. Please login again.", "success")
    return redirect(url_for("login"))


# ---------------- UPDATE PASSWORD ----------------
@app.route("/update_password", methods=["POST"])
def update_password():
    if not login_required():
        return redirect(url_for("login"))

    current_password = request.form["current_password"]
    new_password = request.form["new_password"]

    if not is_strong_password(new_password):
        flash("New password does not meet security requirements.", "error")
        return redirect(url_for("profile"))

    conn = sqlite3.connect("database/health_data.db")
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE email=?", (session["user_email"],))
    row = cursor.fetchone()

    if not row or not check_password_hash(row[0], current_password):
        conn.close()
        flash("Current password incorrect.", "error")
        return redirect(url_for("profile"))

    hashed = generate_password_hash(new_password)
    cursor.execute("UPDATE users SET password=? WHERE email=?", (hashed, session["user_email"]))

    conn.commit()
    conn.close()

    session.clear()
    flash("Password updated successfully. Please login again.", "success")
    return redirect(url_for("login"))


# ---------------- DELETE ACCOUNT ----------------
@app.route("/delete_account", methods=["POST"])
def delete_account():
    if not login_required():
        return redirect(url_for("login"))

    password = request.form["password"]

    conn = sqlite3.connect("database/health_data.db")
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE email=?", (session["user_email"],))
    row = cursor.fetchone()

    if not row or not check_password_hash(row[0], password):
        conn.close()
        flash("Password incorrect. Account not deleted.", "error")
        return redirect(url_for("profile"))

    # Delete predictions first
    cursor.execute("DELETE FROM user_predictions WHERE user_id=?", (session["user_email"],))

    # Delete user
    cursor.execute("DELETE FROM users WHERE email=?", (session["user_email"],))

    conn.commit()
    conn.close()

    session.clear()
    flash("Account deleted successfully.", "success")
    return redirect(url_for("home"))



#----------------forgotpassword----------------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]

        conn = sqlite3.connect("database/health_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            flash("Email not registered.", "error")
            return redirect(url_for("forgot_password"))

        otp = generate_otp()
        session["reset_email"] = email
        session["reset_otp"] = otp
        session["otp_expiry"] = time.time() + 300   # 5 minutes (300 sec)

        msg = Message(
            subject="Your Password Reset OTP",
            recipients=[email],
            body=f"Your OTP for password reset is: {otp}"
        )

        mail.send(msg)

        flash("OTP sent to your email.", "success")
        return redirect(url_for("verify_otp"))

    return render_template("forgot_password.html")



@app.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        user_otp = request.form["otp"]

        if time.time() > session.get("otp_expiry", 0):
            flash("OTP has expired. Please request a new one.", "error")
            return redirect(url_for("forgot_password"))

        if user_otp == session.get("reset_otp"):
            session.pop("reset_otp", None)
            session.pop("otp_expiry", None)
            return redirect(url_for("reset_password"))

        else:
            flash("Invalid OTP.", "error")
            return redirect(url_for("verify_otp"))

    return render_template("verify_otp.html")






@app.route("/resend_otp", methods=["POST"])
def resend_otp():
    email = session.get("reset_email")

    if not email:
        return redirect(url_for("forgot_password"))

    otp = generate_otp()
    session["reset_otp"] = otp
    session["otp_expiry"] = time.time() + 300

    msg = Message(
        subject="Your New OTP",
        recipients=[email],
        body=f"Your new OTP is: {otp}"
    )
    mail.send(msg)

    flash("New OTP sent successfully.", "success")
    return redirect(url_for("verify_otp"))





@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        new_password = request.form["password"]

        if not is_strong_password(new_password):
            flash("Password must meet strength requirements.", "error")
            return redirect(url_for("reset_password"))

        hashed_password = generate_password_hash(new_password)

        conn = sqlite3.connect("database/health_data.db")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password=? WHERE email=?",
            (hashed_password, session.get("reset_email"))
        )
        conn.commit()
        conn.close()

        session.pop("reset_email", None)
        session.pop("reset_otp", None)
        session.pop("otp_expiry", None)


        flash("Password updated successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")


# ---------------- ACCOUNT HISTORY ----------------

from datetime import datetime, timedelta

@app.route("/account_history")
def account_history():
    if not login_required():
        return redirect(url_for("login"))

    search = request.args.get("search", "").strip()
    risk_filter = request.args.get("risk", "").strip()

    conn = sqlite3.connect("database/health_data.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ====================================
    # BASE QUERY
    # ====================================
    base_query = """
        SELECT id, disease_type, probability, risk_level, timestamp
        FROM user_predictions
        WHERE user_id=?
    """
    params = [session["user_email"]]

    if search:
        base_query += " AND disease_type LIKE ?"
        params.append(f"%{search}%")

    if risk_filter:
        base_query += " AND risk_level=?"
        params.append(risk_filter)

    # ====================================
    # TABLE DATA (DESC)
    # ====================================
    table_query = base_query + " ORDER BY timestamp DESC"
    cursor.execute(table_query, params)
    predictions = cursor.fetchall()

    # ====================================
    # CHART DATA (ASC)  ← FIXED HERE
    # ====================================
    chart_query = base_query + " ORDER BY timestamp ASC"
    cursor.execute(chart_query, params)
    chart_rows = cursor.fetchall()

    conn.close()

    # ====================================
    # TABLE TIME CONVERSION
    # ====================================
    updated_predictions = []

    for row in predictions:
        utc_time = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        ist_time = utc_time + timedelta(hours=5, minutes=30)

        updated_predictions.append({
            "id": row["id"],
            "disease_type": row["disease_type"],
            "probability": row["probability"],
            "risk_level": row["risk_level"],
            "timestamp": ist_time.strftime("%d %b %Y, %I:%M %p")
        })

    # ====================================
    # PROGRESS BAR LOGIC
    # ====================================
    disease_latest = {}
    disease_previous = {}

    for row in chart_rows:
        disease = row["disease_type"]
        prob = round(row["probability"] * 100, 2)

        if disease not in disease_latest:
            disease_latest[disease] = prob
            disease_previous[disease] = None
        else:
            disease_previous[disease] = disease_latest[disease]
            disease_latest[disease] = prob

    chart_labels = []
    chart_values = []
    chart_colors = []

    for disease in disease_latest:
        chart_labels.append(disease.capitalize())
        chart_values.append(disease_latest[disease])

        if disease_previous[disease] is None:
            chart_colors.append("#3498db")  # first record
        elif disease_latest[disease] < disease_previous[disease]:
            chart_colors.append("#27ae60")  # improved
        else:
            chart_colors.append("#c0392b")  # worsened

    return render_template(
        "account_history.html",
        predictions=updated_predictions,
        total_predictions=len(updated_predictions),
        search=search,
        risk_filter=risk_filter,
        chart_labels=chart_labels,
        chart_values=chart_values,
        chart_colors=chart_colors
    )





# ---------------- DOWNLOAD SPECIFIC PDF ----------------

@app.route("/download_pdf/<int:prediction_id>")
def download_specific_pdf(prediction_id):

    if not login_required():
        return redirect(url_for("login"))

    conn = sqlite3.connect("database/health_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT disease_type, input_data, probability, risk_level,
               shap_explanation, health_suggestions, timestamp
        FROM user_predictions
        WHERE id=? AND user_id=?
    """, (prediction_id, session["user_email"]))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return "Report not found"

    import json
    from textwrap import wrap

    disease, input_data, probability, risk, shap_explanation, health_suggestions, timestamp = row

    
    utc_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    ist_time = utc_time + timedelta(hours=5, minutes=30)
    formatted_time = ist_time.strftime("%d %b %Y, %I:%M %p")


    input_data = ast.literal_eval(input_data)
    probability = float(probability)

    if health_suggestions:
        health_suggestions = json.loads(health_suggestions)
    else:
        health_suggestions = None

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    # ================= TITLE =================
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(300, y, "AI Healthcare Screening Report")
    y -= 25

    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(300, y, f"Generated on: {formatted_time}")
    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, y, f"User: {session['user_email']}")
    y -= 18

    pdf.drawString(50, y, f"Disease: {disease.capitalize()}")
    y -= 30

    # ================= INPUT PARAMETERS =================
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Input Parameters")
    y -= 20

    pdf.setFont("Helvetica", 11)
    for k, v in input_data.items():
        pdf.drawString(60, y, f"{k.replace('_',' ').title()}: {v}")
        y -= 16

    y -= 20

    # ================= RISK =================
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, f"Risk Level: {risk}")
    y -= 20

    pdf.drawString(50, y, f"Probability: {round(probability*100,2)}%")

    # ================= SHAP EXPLANATION =================
    if shap_explanation:
        y -= 30
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Why this risk level?")
        y -= 18

        pdf.setFont("Helvetica", 11)
        wrapped_text = wrap(shap_explanation, 80)
        text_object = pdf.beginText(60, y)

        for line in wrapped_text:
            text_object.textLine(line)

        pdf.drawText(text_object)
        y -= 14 * len(wrapped_text)

    # ================= HEALTH SUGGESTIONS =================
    if health_suggestions:
        y -= 30
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Personalized Health Suggestions")
        y -= 18

        pdf.setFont("Helvetica", 11)

        title_line = f"{health_suggestions['title']} – {health_suggestions['message']}"
        for line in wrap(title_line, 80):
            pdf.drawString(60, y, line)
            y -= 14

        y -= 10

        for suggestion in health_suggestions["items"]:
            wrapped_lines = wrap(f"- {suggestion}", 75)
            for line in wrapped_lines:
                pdf.drawString(70, y, line)
                y -= 14

        if health_suggestions["level"] == "high":
            y -= 10
            pdf.setFont("Helvetica-Oblique", 9)
            pdf.drawString(
                60, y,
                "Note: These suggestions are supportive and do not replace medical treatment."
            )
            pdf.setFont("Helvetica", 11)

    # ================= DISCLAIMER =================
    y -= 40
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColorRGB(1, 0, 0)
    pdf.drawString(
        50, y,
        "Disclaimer: This is an AI-based screening tool, not a medical diagnosis."
    )

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"prediction_{prediction_id}.pdf",
        mimetype="application/pdf"
    )





@app.route("/delete_prediction/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):

    if not login_required():
        return redirect(url_for("login"))

    conn = sqlite3.connect("database/health_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM user_predictions
        WHERE id=? AND user_id=?
    """, (prediction_id, session["user_email"]))

    conn.commit()
    conn.close()

    flash("Prediction deleted successfully.", "success")
    return redirect(url_for("account_history"))


# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# ======================================================
# LOAD MODELS
# ======================================================

heart_model = joblib.load("models/heart_xgboost_final.pkl")
diabetes_model = joblib.load("models/diabetes_xgboost_final.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
kidney_model = joblib.load("models/kidney_xgboost_final.pkl")
kidney_scaler = joblib.load("models/kidney_scaler.pkl")

diabetes_explainer = shap.TreeExplainer(diabetes_model)
heart_explainer = shap.TreeExplainer(heart_model)
kidney_explainer = shap.TreeExplainer(kidney_model)

# ======================================================
# RISK LOGIC
# ======================================================

def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    return "High Risk"

# ======================================================
# DISEASE SELECTION
# ======================================================

@app.route("/select_disease", methods=["POST"])
def select_disease():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("input_form.html", disease=request.form.get("disease"))

# ======================================================
# PREDICTION
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():
    if not login_required():
        return redirect(url_for("login"))

    user_email = session["user_email"]
    disease = request.form.get("disease")

    shap_data = {}

    # -------- DIABETES --------
    if disease == "diabetes":
        fields = ["pregnancies","glucose","bp","skin","insulin","bmi","dpf","age"]
        X = np.array([[get_float(request.form, f) for f in fields]])
        X_scaled = diabetes_scaler.transform(X)

        prob = float(diabetes_model.predict_proba(X_scaled)[0][1])
        input_data = dict(zip(fields, X[0]))

        shap_data = compute_shap_percentages(diabetes_explainer, X_scaled, fields)

    # -------- HEART --------
    elif disease == "heart":
        fields = ["age","sex","cp","bp","chol","ecg","thalach","exang"]
        X = np.array([[get_float(request.form, f) for f in fields]])

        prob = float(heart_model.predict_proba(X)[0][1])
        input_data = dict(zip(fields, X[0]))

        shap_data = compute_shap_percentages(heart_explainer, X, fields)

    # -------- KIDNEY --------
    else:
        urea, creat, sod, pot, hemo, pcv, sugar, bp, age = [
            get_float(request.form, f)
            for f in ["urea","creatinine","sodium","potassium","hemo","pcv","sugar","bp","age"]
        ]

        engineered = np.array([[ 
            urea + creat,
            sod + pot,
            hemo + pcv,
            sugar,
            bp,
            int(hemo < 12),
            int(pot > 5),
            age
        ]])

        engineered_scaled = kidney_scaler.transform(engineered)
        prob = float(kidney_model.predict_proba(engineered_scaled)[0][1])

        input_data = {
            "urea": urea,
            "creatinine": creat,
            "sodium": sod,
            "potassium": pot,
            "hemo": hemo,
            "pcv": pcv,
            "sugar": sugar,
            "bp": bp,
            "age": age
        }

        kidney_features = [
            "renal_function",
            "electrolyte_balance",
            "blood_health",
            "urine_sugar",
            "blood_pressure",
            "anemia_flag",
            "electrolyte_risk",
            "age"
        ]

        shap_data = compute_shap_percentages(
            kidney_explainer, engineered_scaled, kidney_features
        )

    risk = get_risk_level(prob)

    shap_data = top_5_shap(shap_data)
    shap_explanation = generate_shap_explanation(disease, shap_data)
    health_suggestions = generate_health_suggestions(disease, risk, shap_data)

    
    # ✅ SESSION STORAGE (ONLY ADDITION)
    session["shap_explanation"] = shap_explanation
    session["health_suggestions"] = health_suggestions


    save_prediction(
        user_id=user_email,
        disease_type=disease,
        input_data=input_data,
        probability=prob,
        risk_level=risk,
        model_accuracy=90.0,
        shap_explanation=shap_explanation,
        health_suggestions=health_suggestions
    )


    return render_template(
        "result.html",
        disease=disease,
        risk=risk,
        probability=round(prob * 100, 2),
        user=user_email,
        shap_data=shap_data,
        shap_explanation=shap_explanation,
        health_suggestions=health_suggestions
    )

# ======================================================
# PDF DOWNLOAD
# ======================================================

@app.route("/download_pdf")
def download_pdf():
    if not login_required():
        return redirect(url_for("login"))

    conn = sqlite3.connect("database/health_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT disease_type, input_data, probability, risk_level
        FROM user_predictions
        WHERE user_id=?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (session["user_email"],))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "No report available"

    disease, input_data, probability, risk = row
    input_data = ast.literal_eval(input_data)
    probability = safe_float(probability) 

    # ✅ SESSION READ (ONLY ADDITION)
    shap_explanation = session.get(
        "shap_explanation",
        "Clinical parameters influenced this prediction."
    )

    health_suggestions = session.get("health_suggestions")


    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(300, y, "AI Healthcare Screening Report")
    y -= 25

    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(
        300, y,
        f"Generated on: {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
    )
    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, y, f"User: {session['user_email']}")
    y -= 18
    pdf.drawString(50, y, f"Disease: {disease.capitalize()}")
    y -= 30

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Input Parameters")
    y -= 20

    pdf.setFont("Helvetica", 11)
    for k, v in input_data.items():
        pdf.drawString(60, y, f"{k.replace('_',' ').title()}: {v}")
        y -= 16

    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, f"Risk Level: {risk}")
    y -= 20
    pdf.drawString(50, y, f"Probability: {round(probability*100,2)}%")

    # ================= SHAP EXPLANATION =================
    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Why this risk level?")
    y -= 18

    from textwrap import wrap

    pdf.setFont("Helvetica", 11)

    wrapped_text = wrap(shap_explanation, 80)  # 80 chars per line
    text_object = pdf.beginText(60, y)

    for line in wrapped_text:
        text_object.textLine(line)

    pdf.drawText(text_object)

    # move y down based on number of lines
    y -= 14 * len(wrapped_text)

    # ================= HEALTH SUGGESTIONS =================
    if health_suggestions:
        y -= 30
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Personalized Health Suggestions")
        y -= 18

        pdf.setFont("Helvetica", 11)

        # Title + message
        title_line = f"{health_suggestions['title']} – {health_suggestions['message']}"
        for line in wrap(title_line, 80):
            pdf.drawString(60, y, line)
            y -= 14

        y -= 10

        # Bullet points
        for suggestion in health_suggestions["items"]:
            wrapped_lines = wrap(f"- {suggestion}", 75)
            for line in wrapped_lines:
                pdf.drawString(70, y, line)
                y -= 14

        # High-risk disclaimer
        if health_suggestions["level"] == "high":
            y -= 10
            pdf.setFont("Helvetica-Oblique", 9)
            pdf.drawString(
                60, y,
                "Note: These suggestions are supportive and do not replace medical treatment."
            )
            pdf.setFont("Helvetica", 11)


    # ================= DISCLAIMER =================
    y -= 40
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColorRGB(1, 0, 0)
    pdf.drawString(
        50, y,
        "Disclaimer: This is an AI-based screening tool, not a medical diagnosis."
    )


    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="health_report.pdf",
        mimetype="application/pdf"
    )

# ======================================================
# EXCEL DOWNLOAD
# ======================================================

@app.route("/download_excel")
def download_excel():
    if not login_required():
        return redirect(url_for("login"))

    conn = sqlite3.connect("database/health_data.db")
    df = pd.read_sql_query("""
        SELECT disease_type, input_data, probability, risk_level, timestamp
        FROM user_predictions
        WHERE user_id=?
        ORDER BY timestamp DESC
        LIMIT 1
    """, conn, params=(session["user_email"],))
    conn.close()

    if df.empty:
        return "No prediction found"

    df["input_data"] = df["input_data"].apply(ast.literal_eval)
    df["probability (%)"] = df["probability"].apply(safe_float) * 100

    input_df = pd.json_normalize(df["input_data"])

    final_df = pd.concat(
        [df[["disease_type","risk_level","timestamp"]],
         input_df,
         df[["probability (%)"]]],
        axis=1
    )

    buffer = io.BytesIO()
    final_df.to_excel(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="latest_health_prediction.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ======================================================
# LOGOUT
# ======================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

