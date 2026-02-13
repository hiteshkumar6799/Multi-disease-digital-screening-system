import sqlite3

conn = sqlite3.connect("database/health_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    disease_type TEXT,
    input_data TEXT,
    probability REAL,
    risk_level TEXT,
    shap_explanation TEXT,
    health_suggestions TEXT,
    model_used TEXT,
    model_accuracy REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("✅ Database recreated successfully")
