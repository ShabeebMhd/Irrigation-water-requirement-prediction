import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# =========================================================================
# PHASE 1: PAGE CONFIG (must be the first Streamlit call in the script)
# =========================================================================
st.set_page_config(
    page_title="Sri Lanka Irrigation Predictor",
    page_icon="💧",
    layout="centered",
)

# =========================================================================
# PHASE 1b: VISUAL THEME — light, water-blue, professional
# =========================================================================
# The base palette (background/primary/text colors) lives in
# .streamlit/config.toml — that's the correct place for it, since it's
# read before the app even starts rendering and also themes Streamlit's
# own built-in widgets (buttons, inputs, alerts) automatically. The CSS
# below only adds polish on top: card containers, header banner, and
# metric styling that config.toml can't reach on its own.
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #0EA5E9 0%, #38BDF8 100%);
        padding: 1.75rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.25);
    }
    .app-header h1 {
        color: #FFFFFF !important;
        font-size: 1.6rem;
        margin: 0 0 0.4rem 0;
    }
    .app-header p {
        color: #E0F2FE !important;
        margin: 0;
        font-size: 0.95rem;
    }

    /* Section cards: Streamlit's own st.container(border=True) already
       renders a border derived from the configured theme colors — no
       override needed here, confirmed by inspecting the rendered DOM. */


    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #E0F2FE;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: #0369A1 !important; }
    div[data-testid="stMetricValue"] { color: #0C4A6E !important; }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }

    /* Section headers */
    h3 { color: #0369A1 !important; }
</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = "irrigation_model_artifacts"


# =========================================================================
# PHASE 2: LOAD PRODUCTION ARTIFACTS (cached — loads once, not on every rerun)
# =========================================================================
@st.cache_resource
def load_artifacts():
    """
    Loads every fitted object the pipeline needs. st.cache_resource means
    Streamlit loads these ONCE per server session and reuses them across
    every rerun — without it, every widget click would reload 5 files
    from disk and refit nothing, but still waste time doing so.
    """
    clf        = joblib.load(f"{ARTIFACT_DIR}/stage1_classifier.joblib")
    reg        = joblib.load(f"{ARTIFACT_DIR}/stage2_regressor.joblib")
    scaler     = joblib.load(f"{ARTIFACT_DIR}/stage1_scaler.joblib")
    reg_scaler = joblib.load(f"{ARTIFACT_DIR}/stage2_scaler.joblib")
    ohe        = joblib.load(f"{ARTIFACT_DIR}/onehot_encoder.joblib")
    with open(f"{ARTIFACT_DIR}/metadata.json") as f:
        meta = json.load(f)
    return clf, reg, scaler, reg_scaler, ohe, meta


try:
    clf, reg, scaler, reg_scaler, ohe, meta = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"⚠️ Missing artifact file: {e}\n\n"
        f"Place the entire `{ARTIFACT_DIR}/` folder (created by the notebook's "
        f"serialization cell) in the same directory as this app.py."
    )
    st.stop()


# =========================================================================
# PHASE 3: THE 9 FIELDS THAT ACTUALLY MATTER
# =========================================================================
# Only Season, Climate_Zone, and 7 numeric fields ever reach either model
# (verified: scaler.transform() scales every column independently, so
# every OTHER raw column can be defaulted with a placeholder value below
# and it will not change the prediction by even one floating-point digit).
FIELD_SPECS = {
    "Season": {
        "type": "categorical", "options": ["Maha", "Yala"],
        "label": "Season",
        "hint": "Maha (Oct–Mar, main monsoon) or Yala (Apr–Sep, secondary monsoon)",
    },
    "Climate_Zone": {
        "type": "categorical", "options": ["Dry", "Intermediate", "Wet"],
        "label": "Climate Zone",
        "hint": "Sri Lanka's three agro-climatic zones",
    },
    "Stage_Duration_days": {
        "type": "numeric", "min": 10.0, "max": 65.0, "default": 25.0,
        "label": "Growth Stage Duration (days)",
        "hint": "Length of the current crop growth stage",
    },
    "Tmax_C": {
        "type": "numeric", "min": 20.0, "max": 40.0, "default": 31.0,
        "label": "Max Temperature (°C)",
        "hint": "Average daily maximum temperature for the stage",
    },
    "Humidity_pct": {
        "type": "numeric", "min": 40.0, "max": 100.0, "default": 75.0,
        "label": "Relative Humidity (%)",
        "hint": "Average relative humidity for the stage",
    },
    "Rainfall_mm": {
        "type": "numeric", "min": 0.0, "max": 400.0, "default": 100.0,
        "label": "Rainfall (mm)",
        "hint": "Total rainfall received during the stage",
    },
    "Sunlight_Hours": {
        "type": "numeric", "min": 3.0, "max": 11.0, "default": 7.0,
        "label": "Sunlight (hours/day)",
        "hint": "Average daily sunlight hours for the stage",
    },
    "Reference_ET0_mm_day": {
        "type": "numeric", "min": 2.0, "max": 7.0, "default": 4.2,
        "label": "Reference Evapotranspiration ET₀ (mm/day)",
        "hint": "FAO-56 reference ET₀ — from local weather station or estimate",
    },
    "Crop_Coefficient_Kc": {
        "type": "numeric", "min": 0.3, "max": 1.3, "default": 0.9,
        "label": "Crop Coefficient (Kc)",
        "hint": "FAO-56 Kc value for this crop and growth stage",
    },
}

# These two are shown for the user's own record-keeping / context only.
# The trained models never see them: Crop_Type and Crop_Growth_Stage did not
# survive RFECV feature selection in either stage, so changing these values
# has zero effect on the prediction. Kc already encodes the crop- and
# stage-specific water demand that would otherwise come from these fields.
CONTEXT_FIELD_SPECS = {
    "Crop_Type": {
        "options": ["Big Onion", "Paddy (Rice)", "Tomato", "Green Gram", "Maize", "Chili"],
        "label": "Crop",
        "hint": "For your reference — does not change the prediction (see caption below)",
    },
    "Crop_Growth_Stage": {
        "options": ["Initial", "Development", "Mid-season", "Late-season"],
        "label": "Growth Stage",
        "hint": "For your reference — does not change the prediction (see caption below)",
    },
}


# =========================================================================
# PHASE 4: BUILD A FULL RAW ROW (defaults + the 9 real inputs), THEN PREDICT
# =========================================================================
def build_default_row(raw_columns):
    """
    Every column the scaler was fit on must be present, in the right dtype,
    or pandas/sklearn will throw a shape/column-mismatch error. Columns
    outside FIELD_SPECS never reach either model, so their exact value is
    irrelevant to the prediction — 0 for numeric, first category for text.
    """
    row = {}
    for col in raw_columns:
        if col in ("Net_Irrigation_Requirement_mm", "Field_Cycle_ID"):
            continue
        row[col] = 0.0
    # overwrite the categorical placeholders with *some* valid string,
    # since numeric 0.0 would be the wrong dtype for a text column
    text_defaults = {
        "Crop_Type": "Paddy (Rice)", "Crop_Growth_Stage": "Initial",
        "Soil_Type": "Loamy", "Irrigation_Type": "Drip",
        "Water_Source": "Agro-well", "Mulching_Used": "No",
        "Season": "Maha", "Climate_Zone": "Wet",
    }
    row.update(text_defaults)
    return row


def predict_irrigation(user_inputs: dict):
    """
    Runs the exact same encode -> scale -> classify -> (maybe) regress
    pipeline as predict_hurdle_pipeline() in the training notebook, but
    starting from a single dict of raw feature values instead of a
    pre-encoded test dataframe.
    """
    row = build_default_row(meta["raw_input_columns"])
    row.update(user_inputs)  # the 9 real values override their defaults
    df_in = pd.DataFrame([row])[meta["raw_input_columns"]]

    # binary categorical encode, using the EXACT map fit during training
    for col, mapping in meta["binary_maps"].items():
        df_in[col] = df_in[col].astype(str).map(mapping).fillna(0).astype(int)

    # one-hot encode with the FITTED encoder (unseen categories -> all-zero row, safe)
    ohe_arr = ohe.transform(df_in[meta["multiclass_cols"]].astype(str))
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(meta["multiclass_cols"]), index=df_in.index)
    df_encoded = pd.concat([df_in.drop(columns=meta["multiclass_cols"]), ohe_df], axis=1)

    # Stage 1: scale + classify
    df_scaled_clf = pd.DataFrame(scaler.transform(df_encoded), columns=df_encoded.columns, index=df_encoded.index)
    probability = clf.predict_proba(df_scaled_clf[meta["clf_features"]])[:, 1][0]
    needs_water = probability >= meta["hurdle_threshold"]

    predicted_mm = 0.0
    if needs_water:
        df_scaled_reg = pd.DataFrame(reg_scaler.transform(df_encoded), columns=df_encoded.columns, index=df_encoded.index)
        predicted_mm = float(reg.predict(df_scaled_reg[meta["reg_features"]])[0])
        predicted_mm = max(0.0, predicted_mm)   # ADD THIS — Stage 2 was trained only on y>0 rows,
                                                 # so it has no floor at zero and can extrapolate
                                                 # slightly negative for borderline cases like this one

    return predicted_mm, float(probability), needs_water


# =========================================================================
# PHASE 5: THE UI
# =========================================================================
st.markdown("""
<div class="app-header">
    <h1>💧 Sri Lanka Irrigation Water Requirement Predictor</h1>
    <p>A two-stage Hurdle Model (classify need → predict volume), trained on a
    synthetic FAO-56-physics-grounded dataset. Built as a methodology
    demonstration — see disclaimer below.</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("🌾 Crop Details")
    context_inputs = {}
    ctx_col1, ctx_col2 = st.columns(2)
    for i, (field, spec) in enumerate(CONTEXT_FIELD_SPECS.items()):
        target_col = ctx_col1 if i % 2 == 0 else ctx_col2
        with target_col:
            context_inputs[field] = st.selectbox(spec["label"], spec["options"], help=spec["hint"])
    st.caption(
        "ℹ️ Crop and growth stage are recorded for your reference only. The model's final "
        "5 selected features don't include them — their agronomic effect is already captured "
        "through the Crop Coefficient (Kc) value you enter below, which you should set to "
        "match this crop and stage per FAO-56 tables."
    )

st.write("")

with st.container(border=True):
    st.subheader("🌤️ Field Conditions")
    user_inputs = {}
    col1, col2 = st.columns(2)

    widgets = list(FIELD_SPECS.items())
    for i, (field, spec) in enumerate(widgets):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if spec["type"] == "categorical":
                user_inputs[field] = st.selectbox(spec["label"], spec["options"], help=spec["hint"])
            else:
                user_inputs[field] = st.number_input(
                    spec["label"], min_value=spec["min"], max_value=spec["max"],
                    value=spec["default"], help=spec["hint"],
                )

st.write("")

if st.button("Predict Irrigation Requirement", type="primary", use_container_width=True):
    # Merge in the context selections too — harmless (verified: doesn't change
    # the output), but more honest than silently discarding what the user typed.
    full_inputs = {**user_inputs, **context_inputs}
    predicted_mm, probability, needs_water = predict_irrigation(full_inputs)
    crop_label = f"{context_inputs['Crop_Type']} ({context_inputs['Crop_Growth_Stage']} stage)"

    st.subheader("💧 Result")
    c1, c2 = st.columns(2)
    c1.metric("P(needs irrigation)", f"{probability:.0%}")
    c2.metric("Predicted volume", f"{predicted_mm:.1f} mm" if needs_water else "0.0 mm")

    if needs_water:
        st.success(
            f"**Irrigation recommended for {crop_label}.** Model estimates approximately "
            f"**{predicted_mm:.1f} mm** of net irrigation for this stage "
            f"(gate threshold = {meta['hurdle_threshold']}, confidence {probability:.0%})."
        )
    else:
        st.info(
            f"**No irrigation recommended for {crop_label}** — rainfall and "
            f"conditions appear sufficient (confidence {1 - probability:.0%})."
        )

    with st.expander("How to read this"):
        st.write(
            "Stage 1 estimates the probability this field-stage needs any water at all. "
            f"If that probability clears {meta['hurdle_threshold']} ({int(meta['hurdle_threshold']*100)}%), "
            "Stage 2 estimates how much. The threshold was set below the default 0.50 "
            "deliberately: it trades a small amount of accuracy for meaningfully fewer "
            "missed-irrigation events, since under-watering a crop is costlier than a "
            "little wasted water from a false alarm."
        )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** trained on a synthetic dataset built to be FAO-56-physics-grounded "
    "for methodology validation. This demonstrates a modeling pipeline, not a substitute for "
    "agronomic advice or real-world irrigation scheduling."
)
