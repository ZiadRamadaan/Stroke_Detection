from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import joblib
import pandas as pd
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import xgboost
import lightgbm

app = Flask(__name__)
app.secret_key = 'stroke_secret_key_2026'
CORS(app)

def init_db():
    conn = sqlite3.connect('stroke_system.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_email TEXT,
            age REAL,
            gender TEXT,
            hypertension INTEGER,
            heart_disease INTEGER,
            ever_married TEXT,
            work_type TEXT,
            residence_type TEXT,
            avg_glucose_level REAL,
            bmi REAL,
            smoking_status TEXT,
            risk_percentage REAL,
            prediction_class INTEGER,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

try:
    bundle = joblib.load("stroke_bundle.pkl")
    model = bundle['model']
    preprocessor = bundle['preprocessor']
    threshold = bundle.get('threshold', 0.06)
    print("Model Bundle Loaded Successfully!")
except Exception as e:
    print(f"Error loading bundle: {e}")
    bundle = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    hashed_password = generate_password_hash(password)

    try:
        conn = sqlite3.connect('stroke_system.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Account created successfully! Please sign in."})
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "Email already exists. Please sign in."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    conn = sqlite3.connect('stroke_system.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user[3], password):
        session['logged_in'] = True
        session['user_email'] = email
        session['user_name'] = user[1] 
        return jsonify({"status": "success", "message": "Login successful!"})
    else:
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401

@app.route('/api/predict', methods=['POST'])
def predict_api():
    if not bundle:
        return jsonify({"error": "Model bundle not loaded"}), 500

    try:
        data = request.json
        
        age = float(data.get("age"))
        avg_glucose = float(data.get("avg_glucose_level"))
        bmi = float(data.get("bmi"))
        hypertension = int(data.get("hypertension"))
        heart_disease = int(data.get("heart_disease"))

        input_dict = {
            "gender": str(data.get("gender")).strip(),
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": str(data.get("ever_married")).strip(),
            "work_type": str(data.get("work_type")).strip(),
            "Residence_type": str(data.get("Residence_type")).strip(), 
            "avg_glucose_level": avg_glucose,
            "bmi": bmi,
            "smoking_status": str(data.get("smoking_status")).strip(),
            
            "is_elderly": 1 if age > 60 else 0,
            "high_glucose": 1 if avg_glucose > 140 else 0,
            "is_obese": 1 if bmi > 30 else 0,
            "age_glucose": age * avg_glucose,
            "age_bmi": age * bmi,
            "health_risk_score": hypertension + heart_disease + (1 if bmi > 30 else 0),
            "risk": 1 if (age > 45 and (hypertension == 1 or heart_disease == 1)) else 0
        }

        input_df = pd.DataFrame([input_dict])
        
        X_processed = preprocessor.transform(input_df)
        probability = model.predict_proba(X_processed)[0][1] 
        
        
        prediction = 1 if probability >= threshold else 0
        risk_percentage = round(float(probability * 100), 2)


        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "risk_percentage": risk_percentage,
            "clinical_threshold": threshold
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)