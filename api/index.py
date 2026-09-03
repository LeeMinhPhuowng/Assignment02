import os
import sys
import re
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Inject ZeroMarker into sys.modules and __main__ for unpickling
class ZeroMarker:
    def __init__(self, columns):
        self.columns = list(columns)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        present = [col for col in self.columns if col in frame.columns]
        frame[present] = frame[present].replace(0, np.nan)
        return frame

def convert_invalid_zero_to_nan(data):
    cleaned = data.copy()
    cleaned[["Glucose", "BMI"]] = cleaned[["Glucose", "BMI"]].replace(0, np.nan)
    return cleaned

import __main__
__main__.ZeroMarker = ZeroMarker
__main__.convert_invalid_zero_to_nan = convert_invalid_zero_to_nan

# Cache for loaded models
MODELS = {}

def get_models():
    if not MODELS:
        print("Loading models into memory...")
        # 1. Diabetes
        prep = joblib.load(BASE_DIR / "diabetes" / "model" / "preprocessor.sav")
        if hasattr(prep, "named_steps") and "imputer" in prep.named_steps:
            imp = prep.named_steps["imputer"]
            if not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = np.dtype("float64")
        MODELS["diab_prep"] = prep
        MODELS["diab_model"] = joblib.load(BASE_DIR / "diabetes" / "model" / "decision_tree.sav")
        
        # 2. House Price
        MODELS["house_prep"] = joblib.load(BASE_DIR / "house_price" / "model" / "preprocessor.sav")
        MODELS["house_model"] = joblib.load(BASE_DIR / "house_price" / "model" / "random_forest_regressor.sav")
        
        # 3. Customer Behavior
        MODELS["ecom_vec"] = joblib.load(BASE_DIR / "customer_behavior" / "model" / "tfidf_vectorizer.sav")
        MODELS["ecom_model"] = joblib.load(BASE_DIR / "customer_behavior" / "model" / "logistic_regression.sav")
        print("All 3 models loaded successfully!")
    return MODELS

def clean_nlp_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Hub HTML (ONLY 3 WEB PREDICTION APPS)
HUB_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ Thống Trí Tuệ Nhân Tạo Thông Minh - AI Solutions Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --radius: 18px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 50px 20px;
        }
        .container {
            width: 100%;
            max-width: 1100px;
        }
        .header {
            text-align: center;
            margin-bottom: 48px;
        }
        .badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            border: 1px solid rgba(56, 189, 248, 0.3);
            margin-bottom: 16px;
            letter-spacing: 0.05em;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            font-size: 1.1rem;
            color: var(--text-muted);
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 28px;
            margin-bottom: 40px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 32px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.25s ease;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }
        .card:hover {
            transform: translateY(-6px);
            border-color: #38bdf8;
            box-shadow: 0 20px 35px -10px rgba(56, 189, 248, 0.2);
        }
        .card-icon {
            width: 54px;
            height: 54px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 0.05em;
            color: #ffffff;
            margin-bottom: 20px;
        }
        .icon-diab { background: linear-gradient(135deg, #0284c7, #0d9488); }
        .icon-house { background: linear-gradient(135deg, #d97706, #b45309); }
        .icon-ecom { background: linear-gradient(135deg, #4f46e5, #8b5cf6); }
        
        .card h2 {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: #ffffff;
        }
        .card p {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.55;
            margin-bottom: 24px;
            flex-grow: 1;
        }
        .card-meta {
            font-size: 0.82rem;
            color: #64748b;
            margin-bottom: 20px;
            border-top: 1px solid rgba(255,255,255,0.06);
            padding-top: 12px;
        }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 20px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.92rem;
            text-decoration: none;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .btn-primary {
            background: #38bdf8;
            color: #0f172a;
        }
        .btn-primary:hover {
            background: #7dd3fc;
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
        }
        .footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.88rem;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge">Assignment 02</div>
            <h1>Hệ Thống Trí Tuệ Nhân Tạo Thông Minh</h1>
            <p class="subtitle">Cổng truy cập trải nghiệm trực tuyến 3 ứng dụng Machine Learning độc lập giải quyết bài toán Y tế, Bất động sản và Thương mại điện tử.</p>
        </div>

        <div class="grid">
            <!-- App 1: Diabetes -->
            <div class="card">
                <div>
                    <div class="card-icon icon-diab">MED</div>
                    <h2>MediPredict</h2>
                    <p>Hệ thống hỗ trợ ra quyết định lâm sàng sàng lọc nguy cơ đái tháo đường dựa trên các chỉ số sinh hóa (Glucose, BMI, Độ tuổi, Tiền sử gia đình).</p>
                </div>
                <div>
                    <div class="card-meta">Mô hình: Decision Tree (Recall 72.2%)</div>
                    <a href="/diabetes" class="btn btn-primary">Mở Ứng Dụng Dự Đoán &rarr;</a>
                </div>
            </div>

            <!-- App 2: House Price -->
            <div class="card">
                <div>
                    <div class="card-icon icon-house">EST</div>
                    <h2>EstateValuate</h2>
                    <p>Hệ thống thẩm định giá trị bất động sản tự động tại Việt Nam dựa trên diện tích sàn, số phòng ngủ, phòng tắm, đường vào và mức độ nội thất.</p>
                </div>
                <div>
                    <div class="card-meta">Mô hình: Random Forest (RMSE 1.836 tỷ VNĐ)</div>
                    <a href="/house-price" class="btn btn-primary">Mở Ứng Dụng Định Giá &rarr;</a>
                </div>
            </div>

            <!-- App 3: Customer Behavior -->
            <div class="card">
                <div>
                    <div class="card-icon icon-ecom">NLP</div>
                    <h2>ReviewPulse</h2>
                    <p>Phân hệ xử lý ngôn ngữ tự nhiên (NLP) phân tích cảm xúc và dự báo xu hướng khuyến nghị sản phẩm của khách hàng từ văn bản đánh giá thô.</p>
                </div>
                <div>
                    <div class="card-meta">Mô hình: Logistic Regression Balanced (AUC 0.939)</div>
                    <a href="/customer-behavior" class="btn btn-primary">Mở Ứng Dụng Phân Tích &rarr;</a>
                </div>
            </div>
        </div>

        <div class="footer">
            Lê Minh Phương - B23DCCN663
        </div>
    </div>
</body>
</html>
"""

# Navigation Bar without report link
NAV_BAR_HTML = """
<div style="width: 100%; max-width: 860px; margin-bottom: 20px; display: flex; justify-content: flex-start; align-items: center;">
    <a href="/" style="display: inline-flex; align-items: center; gap: 8px; text-decoration: none; color: #475569; font-weight: 600; font-size: 0.9rem; padding: 8px 16px; background: rgba(255,255,255,0.9); border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
        &larr; Quay lại Hub
    </a>
</div>
"""

def serve_web_page(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "<body>" in content:
        content = content.replace("<body>", f"<body>\n{NAV_BAR_HTML}")
    return render_template_string(content)

# Web UI Routes
@app.route("/")
def index():
    return render_template_string(HUB_HTML)

@app.route("/diabetes")
def web_diabetes():
    return serve_web_page(BASE_DIR / "diabetes" / "web" / "index.html")

@app.route("/house-price")
def web_house_price():
    return serve_web_page(BASE_DIR / "house_price" / "web" / "index.html")

@app.route("/customer-behavior")
def web_customer_behavior():
    return serve_web_page(BASE_DIR / "customer_behavior" / "web" / "index.html")

# Prediction APIs
@app.route("/diabetes/v1/predict", methods=["POST"])
def predict_diabetes():
    try:
        models = get_models()
        payload = request.get_json(force=True)
        glucose = float(payload.get("Glucose", 0))
        bmi = float(payload.get("BMI", 0))
        age = float(payload.get("Age", 0))
        pregnancies = float(payload.get("Pregnancies", 0))
        dpf = float(payload.get("DiabetesPedigreeFunction", 0.5))
        
        if glucose <= 0 or bmi <= 0 or age <= 0:
            return jsonify({"status": "error", "message": "Chỉ số Glucose, BMI và Tuổi phải lớn hơn 0."}), 400
            
        df = pd.DataFrame([{
            "Glucose": glucose, "BMI": bmi, "Age": age,
            "Pregnancies": pregnancies, "DiabetesPedigreeFunction": dpf
        }])
        
        trans = models["diab_prep"].transform(df)
        pred = int(models["diab_model"].predict(trans)[0])
        
        confidence = 85.0
        if hasattr(models["diab_model"], "predict_proba"):
            probs = models["diab_model"].predict_proba(trans)[0]
            confidence = float(probs[pred] * 100)
            
        return jsonify({
            "status": "success",
            "prediction_class": pred,
            "prediction": "Nguy cơ cao mắc đái tháo đường" if pred == 1 else "Nguy cơ thấp mắc đái tháo đường",
            "confidence": confidence,
            "model": "decision_tree",
            "model_label": "Decision Tree (khuyến nghị)"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/house-price/v1/predict", methods=["POST"])
def predict_house_price():
    try:
        models = get_models()
        payload = request.get_json(force=True)
        area = float(payload.get("Area", 0))
        access_road = float(payload.get("Access Road", 0))
        floors = float(payload.get("Floors", 1))
        bedrooms = float(payload.get("Bedrooms", 1))
        bathrooms = float(payload.get("Bathrooms", 1))
        furniture = str(payload.get("Furniture state", "Basic"))
        
        if area <= 0:
            return jsonify({"status": "error", "message": "Diện tích nhà phải lớn hơn 0."}), 400
            
        df = pd.DataFrame([{
            "Area": area, "Access Road": access_road, "Floors": floors,
            "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Furniture state": furniture
        }])
        
        trans = models["house_prep"].transform(df)
        predicted_price = float(models["house_model"].predict(trans)[0])
        predicted_price = max(0.2, round(predicted_price, 2))
        
        return jsonify({
            "status": "success",
            "predicted_price_billion": predicted_price,
            "formatted_price": f"{predicted_price:,.2f} tỷ VNĐ",
            "unit_price_million_per_m2": round((predicted_price * 1000) / area, 2),
            "model": "random_forest_regressor",
            "model_label": "Random Forest Regressor (tối ưu)",
            "disclaimer": "Kết quả định giá dựa trên phân tích tương quan từ dữ liệu giao dịch bất động sản 2024."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/recommendation/v1/predict", methods=["POST"])
def predict_customer_behavior():
    try:
        models = get_models()
        payload = request.get_json(force=True)
        text = payload.get("Review Text", "")
        if not text or not str(text).strip():
            return jsonify({"status": "error", "message": "Nội dung nhận xét không được để trống."}), 400
            
        cleaned = clean_nlp_text(text)
        vec = models["ecom_vec"].transform([cleaned])
        pred = int(models["ecom_model"].predict(vec)[0])
        
        probs = models["ecom_model"].predict_proba(vec)[0]
        prob_positive = float(probs[1] * 100)
        
        return jsonify({
            "status": "success",
            "prediction_class": pred,
            "recommendation": "Khách hàng có xu hướng KHUYẾN NGHỊ sản phẩm" if pred == 1 else "Khách hàng KHÔNG KHUYẾN NGHỊ sản phẩm",
            "recommendation_probability": round(prob_positive, 1),
            "result_level": "positive" if pred == 1 else "negative",
            "model": "logistic_regression",
            "model_label": "Logistic Regression Balanced (khuyến nghị)",
            "word_count": len(cleaned.split())
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/health")
def health_check():
    return jsonify({"status": "healthy", "service": "Intelligent Systems Unified API"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
