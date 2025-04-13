
# VLamax Predictor Web-App
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import streamlit as st

# ---- Daten ----
daten = {
    "Athlet": ["Athlet 1", "Athlet 2", "Athlet 3", "Athlet 4", "Athlet 5", "Athlet 6"],
    "Gewicht": [77.4, 78.5, 57.8, 68, 68, 56],
    "Dauer": [19, 19, 20, 18, 20, 23],
    "Watt Durchschnitt": [649, 649, 664, 664, 719, 405],
    "VLamax": [0.42, 0.43, 0.61, 0.51, 0.56, 0.39]
}

# ---- Modelltraining ----
df = pd.DataFrame(daten)
X = df[["Gewicht", "Dauer", "Watt Durchschnitt"]]
y = df["VLamax"]
modell = LinearRegression().fit(X, y)

# ---- Streamlit UI ----
st.title("VLamax Predictor")
st.write("Basierend auf Sprintleistung, Gewicht und Dauer")

gewicht = st.number_input("Gewicht (kg)", min_value=30.0, max_value=120.0, value=70.0, step=0.1)
dauer = st.number_input("Sprintdauer (s)", min_value=5, max_value=30, value=20, step=1)
watt = st.number_input("Durchschnittliche Leistung (W)", min_value=200, max_value=1500, value=650, step=10)

if st.button("VLamax berechnen"):
    prediction = modell.predict([[gewicht, dauer, watt]])[0]
    st.success(f"Geschätzte VLamax: {prediction:.3f} mmol/l/s")

    # Balkenanzeige
    st.progress(min(max(prediction, 0.0), 1.0))

# ---- Modell speichern (optional) ----
joblib.dump(modell, "vlamax_model.joblib")
