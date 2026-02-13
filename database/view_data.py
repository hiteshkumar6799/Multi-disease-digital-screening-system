import sqlite3

conn = sqlite3.connect("database/health_data.db")
cursor = conn.cursor()

print("USERS TABLE:")
cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)

print("\nPREDICTIONS TABLE:")
cursor.execute("SELECT * FROM user_predictions")
for row in cursor.fetchall():
    print(row)

conn.close()
