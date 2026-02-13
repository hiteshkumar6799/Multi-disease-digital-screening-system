import sqlite3
import json

DB_PATH = "database/health_data.db"

def save_prediction(
    user_id,
    disease_type,
    input_data,
    probability,
    risk_level,
    shap_explanation,
    health_suggestions,
    model_accuracy
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO user_predictions (
        user_id,
        disease_type,
        input_data,
        probability,
        risk_level,
        shap_explanation,
        health_suggestions,
        model_used,
        model_accuracy
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        disease_type,
        json.dumps(input_data),
        probability,
        risk_level,
        shap_explanation,
        json.dumps(health_suggestions),
        "XGBoost",
        model_accuracy
    ))

    conn.commit()
    conn.close()



if __name__ == "__main__":
    sample_input = {
        "age": 52,
        "bp": 140,
        "glucose": 180
    }

    sample_suggestions = {
        "level": "high",
        "title": "High Risk Detected",
        "message": "Consult doctor immediately",
        "items": ["Reduce sugar", "Exercise daily"]
    }

    save_prediction(
        user_id="user_001",
        disease_type="diabetes",
        input_data=sample_input,
        probability=0.87,
        risk_level="High Risk",
        shap_explanation="High glucose and age are major contributors.",
        health_suggestions=sample_suggestions,
        model_accuracy=90.16
    )

    print("Sample prediction saved to database")
