# VLamax Predictor Web-App
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import streamlit as st
import streamlit.components.v1 as components

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
st.image("https://raw.githubusercontent.com/360coachinglab/vlamax-app/main/360coachinglab%20gross.png", width=300)
st.title("VLamax Predictor")
st.write("Basierend auf Sprintleistung, Gewicht und Dauer")

gewicht = st.number_input("Gewicht (kg)", min_value=30.0, max_value=120.0, value=70.0, step=0.1)
dauer = st.number_input("Sprintdauer (s)", min_value=5, max_value=30, value=20, step=1)
watt = st.number_input("Durchschnittliche Leistung (W)", min_value=200, max_value=1500, value=650, step=10)

if st.button("VLamax berechnen"):
    prediction = modell.predict([[gewicht, dauer, watt]])[0]
    st.success(f"Geschätzte VLamax: {prediction:.3f} mmol/l/s")

    # Batterie-Anzeige als HTML + CSS
    components.html(f"""
    <div style='display: flex; flex-direction: column; align-items: center;'>
        <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>VLamax Batterie</div>
        <div style='width: 60px; height: 200px; border: 2px solid #333; border-radius: 6px; position: relative; background: #e0e0e0;'>
            <div style='width: 100%; background: #4CAF50; position: absolute; bottom: 0; height: {min(max(prediction, 0), 1) * 100:.1f}%'></div>
        </div>
        <div style='margin-top: 8px; font-weight: bold;'>{prediction:.2f} mmol/l/s</div>
    </div>
    """, height=280)

# ---- Modell speichern (optional) ----
joblib.dump(modell, "vlamax_model.joblib")
