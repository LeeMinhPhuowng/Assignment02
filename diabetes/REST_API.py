import math
import os
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from zero_marker import ZeroMarker  # noqa: F401  (needed when unpickling preprocessor)
from flask import Flask, jsonify, request, send_from_directory
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

KNOWLEDGE_GROUPS = {
    "HAS_DIET_ADVICE": "Chế độ ăn",
    "HAS_LIFESTYLE_ACTION": "Lối sống và vận động",
    "HAS_DRUG_INFO": "Thông tin thuốc",
    "HAS_COMPLICATION": "Biến chứng cần lưu ý",
}

FEATURE_COLUMNS = [
    "Glucose",
    "BMI",
    "Age",
    "Pregnancies",
    "DiabetesPedigreeFunction",
]
INVALID_ZERO_COLUMNS = ["Glucose", "BMI"]

MODEL_OPTIONS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "filename": "logistic_regression.sav",
    },
    "knn": {"label": "K-Nearest Neighbors", "filename": "knn.sav"},
    "linear_svm": {"label": "SVM Linear", "filename": "linear_svm.sav"},
    "rbf_svm": {"label": "SVM RBF", "filename": "rbf_svm.sav"},
    "decision_tree": {
        "label": "Decision Tree (recommended)",
        "filename": "decision_tree.sav",
    },
    "random_forest": {
        "label": "Random Forest",
        "filename": "random_forest.sav",
    },
}


def convert_invalid_zero_to_nan(data):
    cleaned = data.copy()
    cleaned[INVALID_ZERO_COLUMNS] = cleaned[INVALID_ZERO_COLUMNS].replace(0, pd.NA)
    return cleaned

import sys
sys.modules['__main__'].convert_invalid_zero_to_nan = convert_invalid_zero_to_nan

def load_pickle(path):
    return joblib.load(path)


def load_models():
    missing_files = [
        config["filename"]
        for config in MODEL_OPTIONS.values()
        if not (MODEL_DIR / config["filename"]).exists()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing model files in {MODEL_DIR}: {missing}")

    return {
        model_id: load_pickle(MODEL_DIR / config["filename"])
        for model_id, config in MODEL_OPTIONS.items()
    }


if not (MODEL_DIR / "preprocessor.sav").exists():
    raise FileNotFoundError(f"Missing preprocessor file: {MODEL_DIR / 'preprocessor.sav'}")

preprocessor = load_pickle(MODEL_DIR / "preprocessor.sav")
loaded_models = load_models()
neo4j_driver = (
    GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    if NEO4J_PASSWORD
    else None
)

app = Flask(__name__)
if neo4j_driver is None:
    app.logger.warning("Neo4j knowledge graph is disabled: NEO4J_PASSWORD is not configured.")
else:
    app.logger.info("Neo4j knowledge graph is configured for %s.", NEO4J_URI)


def get_diabetes_knowledge():
    """Read Neo4j advisory knowledge only after a high-risk prediction."""
    if neo4j_driver is None:
        return [], "Neo4j password has not been configured."

    query = """
    MATCH (d:Disease {id: 'diabetes'})
    -[relation:HAS_DIET_ADVICE|HAS_LIFESTYLE_ACTION|HAS_DRUG_INFO|HAS_COMPLICATION]
    ->(item)
    OPTIONAL MATCH (item)-[:SOURCED_FROM]->(article:Article)
    RETURN type(relation) AS relation_type,
           coalesce(item.title, item.name) AS title,
           item.content AS content,
           item.duration AS duration,
           item.frequency AS frequency,
           collect(DISTINCT {title: article.title, url: article.url}) AS sources
    ORDER BY relation_type, title
    """

    try:
        grouped_items = defaultdict(list)
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            for record in session.run(query):
                sources = [
                    source for source in record["sources"]
                    if source.get("title") and source.get("url")
                ]
                group = KNOWLEDGE_GROUPS.get(record["relation_type"], record["relation_type"])
                grouped_items[group].append(
                    {
                        "title": record["title"],
                        "content": record["content"],
                        "duration": record["duration"],
                        "frequency": record["frequency"],
                        "sources": sources,
                    }
                )
        return [{"title": group, "items": items} for group, items in grouped_items.items()], None
    except Exception:
        app.logger.exception("Unable to query Neo4j knowledge graph")
        return [], "Không thể tải kiến thức Neo4j lúc này."


def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, "Body must be a JSON object."

    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in payload]
    if missing_features:
        return None, f"Missing required fields: {', '.join(missing_features)}"

    try:
        values = {feature: float(payload[feature]) for feature in FEATURE_COLUMNS}
    except (TypeError, ValueError):
        return None, "All feature values must be numeric."

    if not all(math.isfinite(value) for value in values.values()):
        return None, "All feature values must be finite numbers."
    if values["Glucose"] <= 0 or values["BMI"] <= 0:
        return None, "Glucose and BMI must be greater than 0."
    if values["Age"] <= 0:
        return None, "Age must be greater than 0."
    if values["Pregnancies"] < 0:
        return None, "Pregnancies cannot be negative."
    if values["DiabetesPedigreeFunction"] < 0:
        return None, "DiabetesPedigreeFunction cannot be negative."

    return values, None


@app.get("/")
@app.get("/web")
def web_client():
    return send_from_directory(APP_DIR / "web", "index.html")


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        models=list(MODEL_OPTIONS),
        feature_columns=FEATURE_COLUMNS,
        neo4j_configured=neo4j_driver is not None,
    )


@app.post("/diabetes/v1/predict")
def predict():
    payload = request.get_json(silent=True)
    values, error = validate_payload(payload)
    if error:
        return jsonify(error=error), 400

    model_id = payload.get("model", "decision_tree")
    if model_id not in loaded_models:
        return jsonify(
            error="Invalid model.",
            available_models=list(MODEL_OPTIONS),
        ), 400

    raw_input = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    processed_input = preprocessor.transform(raw_input)
    selected_model = loaded_models[model_id]

    prediction_class = int(selected_model.predict(processed_input)[0])
    probability = None
    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(processed_input)[0]
        probability = round(float(probabilities[prediction_class]) * 100, 2)

    # Neo4j does not influence the prediction. It only provides advisory
    # knowledge for a high-risk result after the ML model has predicted.
    knowledge = []
    knowledge_error = None
    if prediction_class == 1:
        knowledge, knowledge_error = get_diabetes_knowledge()
        if knowledge_error:
            app.logger.warning("High-risk prediction: Neo4j knowledge unavailable (%s)", knowledge_error)
        else:
            app.logger.info("High-risk prediction: returned %d Neo4j knowledge groups.", len(knowledge))

    return jsonify(
        model=model_id,
        model_label=MODEL_OPTIONS[model_id]["label"],
        prediction_class=prediction_class,
        risk_level=("high" if prediction_class == 1 else "low"),
        prediction=(
            "Nguy cơ cao tiểu đường theo mô hình"
            if prediction_class == 1
            else "Nguy cơ thấp tiểu đường theo mô hình"
        ),
        confidence=probability,
        knowledge=knowledge,
        knowledge_error=knowledge_error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
