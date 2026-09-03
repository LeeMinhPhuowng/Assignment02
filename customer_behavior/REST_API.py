from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

from text_prep import normalize_review


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
DEFAULT_MODEL = "logistic_regression"

MODEL_OPTIONS = {
    "logistic_regression": {"label": "Logistic Regression (khuyến nghị)", "filename": "logistic_regression.sav"},
    "linear_svm": {"label": "Linear SVM", "filename": "linear_svm.sav"},
    "multinomial_naive_bayes": {"label": "Multinomial Naive Bayes", "filename": "multinomial_naive_bayes.sav"},
    "sgd_classifier": {"label": "SGD Classifier", "filename": "sgd_classifier.sav"},
    "decision_tree": {"label": "Decision Tree", "filename": "decision_tree.sav"},
    "random_forest": {"label": "Random Forest", "filename": "random_forest.sav"},
}


def load_pickle(path):
    return joblib.load(path)


def load_artifacts():
    required = ["tfidf_vectorizer.sav"] + [config["filename"] for config in MODEL_OPTIONS.values()]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu artifact trong {MODEL_DIR}: {', '.join(missing)}. "
            "Hãy chạy notebook Customer_behavior.ipynb trước."
        )
    vectorizer = load_pickle(MODEL_DIR / "tfidf_vectorizer.sav")
    models = {
        model_id: load_pickle(MODEL_DIR / config["filename"])
        for model_id, config in MODEL_OPTIONS.items()
    }
    return vectorizer, models


tfidf_vectorizer, loaded_models = load_artifacts()
app = Flask(__name__)


def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, None, "Body phải là JSON object."
    if "Review Text" not in payload:
        return None, None, "Thiếu trường Review Text."

    title = str(payload.get("Title", "") or "").strip()
    review_text = str(payload["Review Text"] or "").strip()
    if not review_text:
        return None, None, "Review Text không được để trống."

    model_id = payload.get("model", DEFAULT_MODEL)
    if model_id not in loaded_models:
        return None, None, f"Model không hợp lệ. Chọn một trong: {', '.join(MODEL_OPTIONS)}."

    cleaned_text = normalize_review(f"{title} {review_text}")
    if not cleaned_text:
        return None, None, "Nội dung review không chứa ký tự chữ cái hợp lệ."
    return cleaned_text, model_id, None


@app.get("/")
@app.get("/web")
def web_client():
    return send_from_directory(APP_DIR / "web", "index.html")


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        default_model=DEFAULT_MODEL,
        input_fields=["Title (optional)", "Review Text (required)"],
        models={model_id: config["label"] for model_id, config in MODEL_OPTIONS.items()},
    )


@app.post("/recommendation/v1/predict")
def predict():
    cleaned_text, model_id, error = validate_payload(request.get_json(silent=True))
    if error:
        return jsonify(error=error), 400

    text_vector = tfidf_vectorizer.transform([cleaned_text])
    model = loaded_models[model_id]
    prediction_class = int(model.predict(text_vector)[0])

    probability_recommend = None
    if hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(1)
        probability_recommend = round(float(model.predict_proba(text_vector)[0][class_index]) * 100, 2)

    return jsonify(
        model=model_id,
        model_label=MODEL_OPTIONS[model_id]["label"],
        prediction_class=prediction_class,
        recommendation=("Có khả năng khuyến nghị sản phẩm" if prediction_class == 1 else "Có khả năng không khuyến nghị sản phẩm"),
        result_level=("positive" if prediction_class == 1 else "negative"),
        recommendation_probability=probability_recommend,
        message="Kết quả được dự đoán từ nội dung review và chỉ mang tính tham khảo.",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
