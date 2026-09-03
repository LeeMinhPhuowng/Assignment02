import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

NUMERIC_FEATURES = ["Area", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
CATEGORICAL_FEATURES = ["Furniture state"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_MODEL = "random_forest_regressor"
FURNITURE_STATES = {"Basic", "Full"}

MODEL_OPTIONS = {
    "linear_regression": {"label": "Linear Regression", "filename": "linear_regression.sav"},
    "ridge_regression": {"label": "Ridge Regression", "filename": "ridge_regression.sav"},
    "decision_tree_regressor": {"label": "Decision Tree Regressor", "filename": "decision_tree_regressor.sav"},
    "random_forest_regressor": {"label": "Random Forest Regressor (khuyến nghị)", "filename": "random_forest_regressor.sav"},
    "gradient_boosting_regressor": {"label": "Gradient Boosting Regressor", "filename": "gradient_boosting_regressor.sav"},
}

def load_pickle(path):
    return joblib.load(path)

def load_models():
    files = ["preprocessor.sav"] + [config["filename"] for config in MODEL_OPTIONS.values()]
    missing = [filename for filename in files if not (MODEL_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu file model trong {MODEL_DIR}: {', '.join(missing)}")
    return (
        load_pickle(MODEL_DIR / "preprocessor.sav"),
        {model_id: load_pickle(MODEL_DIR / config["filename"]) for model_id, config in MODEL_OPTIONS.items()},
    )

preprocessor, loaded_models = load_models()
app = Flask(__name__)

def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, None, "Body phải là một JSON object."
    missing = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing:
        return None, None, f"Thiếu trường bắt buộc: {', '.join(missing)}."
    try:
        numeric_values = {field: float(payload[field]) for field in NUMERIC_FEATURES}
    except (TypeError, ValueError):
        return None, None, "Các feature số phải là số hợp lệ."
    if not all(math.isfinite(value) for value in numeric_values.values()):
        return None, None, "Các feature số phải là giá trị hữu hạn."
    if any(value <= 0 for value in numeric_values.values()):
        return None, None, "Area, Access Road, Floors, Bedrooms và Bathrooms phải lớn hơn 0."

    furniture_state = str(payload["Furniture state"]).strip()
    if furniture_state not in FURNITURE_STATES:
        return None, None, "Furniture state chỉ nhận Basic hoặc Full."
    model_id = payload.get("model", DEFAULT_MODEL)
    if model_id not in loaded_models:
        return None, None, f"Model không hợp lệ. Chọn một trong: {', '.join(MODEL_OPTIONS)}."
    return {**numeric_values, "Furniture state": furniture_state}, model_id, None

@app.get("/")
@app.get("/web")
def web_client():
    return send_from_directory(APP_DIR / "web", "index.html")

@app.get("/health")
def health():
    return jsonify(
        status="ok",
        default_model=DEFAULT_MODEL,
        feature_columns=FEATURE_COLUMNS,
        models={model_id: config["label"] for model_id, config in MODEL_OPTIONS.items()},
    )

@app.post("/house-price/v1/predict")
def predict():
    values, model_id, error = validate_payload(request.get_json(silent=True))
    if error:
        return jsonify(error=error), 400
    raw_input = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = float(loaded_models[model_id].predict(preprocessor.transform(raw_input))[0])
    return jsonify(
        model=model_id,
        model_label=MODEL_OPTIONS[model_id]["label"],
        predicted_price=round(prediction, 4),
        predicted_price_display=f"{prediction:.2f} tỷ VNĐ",
        message="Giá dự đoán chỉ mang tính tham khảo, không phải giá thẩm định chính thức.",
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
