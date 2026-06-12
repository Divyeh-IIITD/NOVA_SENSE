from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    st.error("Missing dependency: shap. Install the project requirements and restart the app.")
    st.stop()

from lightgbm import LGBMClassifier


APP_TITLE = "NOVA Food Processing Explorer"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Data" / "Numerical" / "Data" / "65_Nutrients_Data.csv"
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "numerical_65_lgbm" / "strat" / "lgbm_numerical_65_stratified.joblib",
    PROJECT_ROOT / "models" / "numerical_65_lgbm" / "smote_strat" / "lgbm_numerical_65_smote_stratified.joblib",
    PROJECT_ROOT / "models" / "numerical_65_lgbm" / "smote" / "lgbm_numerical_65_smote.joblib",
]

FEATURE_GROUPS = [
    ("Macronutrients & Energy", [
        "Protein",
        "Total Fat",
        "Carbohydrate",
        "Energy",
        "Alcohol",
        "Water",
        "Sugars, total",
        "Fiber, total dietary",
    ]),
    ("Other Compounds", [
        "Caffeine",
        "Theobromine",
    ]),
    ("Minerals", [
        "Calcium",
        "Iron",
        "Magnesium",
        "Phosphorus",
        "Potassium",
        "Sodium",
        "Zinc",
        "Copper",
        "Selenium",
    ]),
    ("Vitamins & Micronutrients", [
        "Retinol",
        "Vitamin A, RAE",
        "Carotene, beta",
        "Carotene, alpha",
        "Vitamin E (alpha-tocopherol)",
        "Vitamin D (D2 + D3)",
        "Cryptoxanthin, beta",
        "Lycopene",
        "Lutein + zeaxanthin",
        "Vitamin C",
        "Thiamin",
        "Riboflavin",
        "Niacin",
        "Vitamin B-6",
        "Folate, total",
        "Vitamin B-12",
        "Choline, total",
        "Vitamin K (phylloquinone)",
        "Folic acid",
        "Folate, food",
        "Folate, DFE",
        "Vitamin E, added",
        "Vitamin B-12, added",
    ]),
    ("Lipids & Fatty Acids", [
        "Cholesterol",
        "Fatty acids, total saturated",
        "4:0",
        "6:0",
        "8:0",
        "10:0",
        "12:0",
        "14:0",
        "16:0",
        "18:0",
        "18:1",
        "18:2",
        "18:3",
        "20:4",
        "22:6 n-3",
        "16:1",
        "18:4",
        "20:1",
        "20:5 n-3",
        "22:1",
        "22:5 n-3",
        "Fatty acids, total monounsaturated",
        "Fatty acids, total polyunsaturated",
    ]),
]


def sanitize_feature_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


@st.cache_data(show_spinner=False)
def load_feature_schema() -> list[dict[str, str]]:
    header = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    feature_names = header[:-1]
    feature_lookup = {sanitize_feature_name(name): name for name in feature_names}

    schema: list[dict[str, str]] = []
    for section_name, section_features in FEATURE_GROUPS:
        for feature_name in section_features:
            key = sanitize_feature_name(feature_name)
            if key not in feature_lookup:
                raise KeyError(f"Feature '{feature_name}' is missing from the dataset schema.")
            schema.append(
                {
                    "section": section_name,
                    "display_name": feature_name,
                    "feature_name": feature_lookup[key],
                    "feature_key": key,
                }
            )

    if len(schema) != 65:
        raise ValueError(f"Expected 65 features, found {len(schema)}.")

    return schema


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[LGBMClassifier, shap.TreeExplainer, list[dict[str, str]]]:
    model_path = None
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError("No LightGBM joblib file was found in the expected model directories.")

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    feature_schema = load_feature_schema()
    return model, explainer, feature_schema


@st.cache_data(show_spinner=False)
def load_demo_input_values(feature_schema: list[dict[str, str]], sample_index: int = 0) -> dict[str, float]:
    data = pd.read_csv(DATA_PATH)
    row = data.iloc[sample_index]
    return {
        item["feature_key"]: float(row[item["feature_name"]])
        for item in feature_schema
    }


def preprocess_input(input_values: dict[str, float], feature_schema: list[dict[str, str]]) -> pd.DataFrame:
    ordered_values = [float(input_values[item["feature_key"]]) for item in feature_schema]
    return pd.DataFrame([ordered_values], columns=[item["feature_key"] for item in feature_schema])


def predict_nova(model: LGBMClassifier, input_frame: pd.DataFrame) -> dict[str, object]:
    probabilities = model.predict_proba(input_frame)[0]
    class_positions = np.arange(len(model.classes_))
    predicted_index = int(np.argmax(probabilities))
    predicted_class = int(model.classes_[predicted_index])

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities,
        "class_positions": class_positions,
        "classes": [int(class_label) for class_label in model.classes_],
    }


def generate_shap_explanation(
    model: LGBMClassifier,
    explainer: shap.TreeExplainer,
    input_frame: pd.DataFrame,
    feature_schema: list[dict[str, str]],
) -> dict[str, object]:
    explanation = explainer(input_frame)
    probabilities = model.predict_proba(input_frame)[0]
    predicted_index = int(np.argmax(probabilities))

    display_name_lookup = {item["feature_key"]: item["display_name"] for item in feature_schema}

    if explanation.values.ndim == 3:
        class_values = explanation.values[0, :, predicted_index]
        base_value = explanation.base_values[0, predicted_index]
    else:
        class_values = explanation.values[0]
        base_value = explanation.base_values[0]

    feature_names = input_frame.columns.tolist()
    feature_values = input_frame.iloc[0].to_numpy(dtype=float)
    shap_explanation = shap.Explanation(
        values=class_values,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names,
    )

    importance_frame = pd.DataFrame(
        {
            "feature": [display_name_lookup.get(name, name) for name in feature_names],
            "shap_value": class_values,
            "abs_shap_value": np.abs(class_values),
            "input_value": feature_values,
        }
    ).sort_values("abs_shap_value", ascending=False)

    top_five = importance_frame.head(5).copy()
    top_five["direction"] = np.where(top_five["shap_value"] >= 0, "Positive", "Negative")
    top_five["shap_value"] = top_five["shap_value"].round(4)
    top_five["abs_shap_value"] = top_five["abs_shap_value"].round(4)
    top_five["input_value"] = top_five["input_value"].round(4)

    return {
        "explanation": shap_explanation,
        "importance_frame": importance_frame,
        "top_five": top_five,
        "predicted_index": predicted_index,
    }


def make_probability_figure(probabilities: np.ndarray, classes: list[int]) -> go.Figure:
    colors = ["#2563eb" if probability == float(np.max(probabilities)) else "#94a3b8" for probability in probabilities]
    figure = go.Figure(
        data=[
            go.Bar(
                x=[f"Class {class_label}" for class_label in classes],
                y=np.round(probabilities * 100, 2),
                marker_color=colors,
                text=[f"{value:.1f}%" for value in np.round(probabilities * 100, 2)],
                textposition="auto",
            )
        ]
    )
    figure.update_layout(
        title="Prediction Probabilities",
        xaxis_title="NOVA Class",
        yaxis_title="Probability (%)",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def make_shap_importance_figure(importance_frame: pd.DataFrame, max_features: int = 15) -> go.Figure:
    top_features = importance_frame.head(max_features).iloc[::-1]
    figure = go.Figure(
        data=[
            go.Bar(
                x=top_features["abs_shap_value"],
                y=top_features["feature"],
                orientation="h",
                marker_color="#0f766e",
                text=[f"{value:.3f}" for value in top_features["abs_shap_value"]],
                textposition="outside",
            )
        ]
    )
    figure.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Mean absolute SHAP contribution",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(520, 28 * len(top_features) + 180),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def render_input_form(feature_schema: list[dict[str, str]]) -> dict[str, float]:
    st.subheader("Nutrient Inputs")
    st.caption("Enter the 65 nutrient values and submit to generate a NOVA class prediction and SHAP explanation.")

    input_values: dict[str, float] = {}

    with st.form("nutrient_input_form"):
        grouped_features: dict[str, list[dict[str, str]]] = {}
        for item in feature_schema:
            grouped_features.setdefault(item["section"], []).append(item)

        for section_name, items in grouped_features.items():
            with st.expander(section_name, expanded=(section_name == "Macronutrients & Energy")):
                columns = st.columns(2)
                for index, item in enumerate(items):
                    column = columns[index % len(columns)]
                    default_value = float(st.session_state.get(item["feature_key"], 0.0))
                    with column:
                        input_values[item["feature_key"]] = st.number_input(
                            item["display_name"],
                            min_value=0.0,
                            value=default_value,
                            step=0.01,
                            format="%.4f",
                            key=item["feature_key"],
                        )

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return {}

    return input_values


def display_results(
    prediction_result: dict[str, object],
    shap_result: dict[str, object],
    input_frame: pd.DataFrame,
) -> None:
    predicted_class = int(prediction_result["predicted_class"])
    confidence = float(prediction_result["confidence"])
    probabilities = np.asarray(prediction_result["probabilities"], dtype=float)
    classes = prediction_result["classes"]
    top_five = shap_result["top_five"].copy()
    explanation = shap_result["explanation"]
    importance_frame = shap_result["importance_frame"]

    st.divider()
    st.subheader("Prediction Results")

    metrics = st.columns(2)
    metrics[0].metric("Predicted NOVA Class", f"{predicted_class}")
    metrics[1].metric("Confidence", f"{confidence * 100:.1f}%")

    st.plotly_chart(make_probability_figure(probabilities, classes), use_container_width=True)

    st.markdown("### Top Contributors")
    st.dataframe(
        top_five[["feature", "shap_value", "direction", "input_value"]].rename(
            columns={
                "feature": "Nutrient",
                "shap_value": "SHAP Value",
                "direction": "Direction",
                "input_value": "Input Value",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### SHAP Waterfall")
    st.caption("Positive SHAP values push the prediction toward the selected NOVA class; negative values pull it away.")
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.markdown("### SHAP Feature Importance")
    st.plotly_chart(make_shap_importance_figure(importance_frame), use_container_width=True)

    st.markdown("### SHAP Summary")
    summary_text = (
        f"The predicted class is NOVA {predicted_class} with {confidence * 100:.1f}% confidence. "
        f"The strongest contributor is {top_five.iloc[0]['feature']} with a SHAP value of {top_five.iloc[0]['shap_value']:.4f}."
    )
    st.info(summary_text)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
            .app-shell {
                background: linear-gradient(135deg, rgba(15,23,42,1) 0%, rgba(30,41,59,1) 45%, rgba(12,74,110,1) 100%);
                color: white;
                padding: 1.4rem 1.6rem;
                border-radius: 1rem;
                margin-bottom: 1rem;
            }
            .app-shell h1, .app-shell p {
                margin: 0;
            }
        </style>
        <div class="app-shell">
            <h1>NOVA Food Processing Explorer</h1>
            <p>Enter nutrient values, predict the NOVA class, and inspect the model with SHAP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, explainer, feature_schema = load_model()
    except Exception as exc:  # pragma: no cover - runtime guard
        st.error(f"Unable to load the model or feature schema: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Model Info")
        st.write(f"Loaded model: {Path(MODEL_CANDIDATES[0]).name}")
        st.write("Target classes: 1, 2, 3, 4")
        st.write("Feature count: 65")
        st.write(f"Schema source: {DATA_PATH.name}")

        if st.button("Load demo sample", use_container_width=True):
            demo_values = load_demo_input_values(feature_schema, sample_index=0)
            for key, value in demo_values.items():
                st.session_state[key] = value
            st.session_state["demo_loaded"] = True
            st.rerun()

        if st.session_state.get("demo_loaded"):
            st.success("Demo sample loaded into the form. Click Predict to run the full explanation.")

    user_input = render_input_form(feature_schema)
    if not user_input:
        return

    input_frame = preprocess_input(user_input, feature_schema)
    prediction_result = predict_nova(model, input_frame)
    shap_result = generate_shap_explanation(model, explainer, input_frame, feature_schema)
    display_results(prediction_result, shap_result, input_frame)


if __name__ == "__main__":
    main()