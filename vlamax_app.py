# VLamax Predictor Web-App mit erweitertem Modell
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import streamlit as st
import streamlit.components.v1 as components

# ---- Web App Layout ----
st.image("https://raw.githubusercontent.com/360coachinglab/vlamax-app/main/360coachinglab%20gross.png", width=300)
st.title("VLamax Auswertung")
st.write("Erweiterte Schätzung basierend auf Geschlecht, Körperkomposition und Sprintleistung")

# ---- Benutzereingaben ----
geschlecht = st.selectbox("Geschlecht", ["Mann", "Frau"])
gewicht = st.number_input("Gewicht (kg)", min_value=30.0, max_value=120.0, value=70.0, step=0.1)
koerperfett = st.number_input("Körperfett (%)", min_value=5.0, max_value=50.0, value=15.0, step=0.1)
groesse = st.number_input("Grösse (cm)", min_value=140, max_value=210, value=175, step=1)
alter = st.number_input("Alter", min_value=10, max_value=80, value=25, step=1)
dauer = st.number_input("Sprintdauer (s)", min_value=5, max_value=30, value=20, step=1)
watt_avg = st.number_input("Durchschnittliche Leistung (W)", min_value=200, max_value=1500, value=650, step=10)
watt_peak = st.number_input("Spitzenleistung (W)", min_value=200, max_value=2000, value=900, step=10)

# ---- Feature-Berechnung ----
ffm = gewicht * (1 - koerperfett / 100)
geschlecht_code = 1 if geschlecht.lower() == "frau" else 0

# ---- Trainingsdaten ----
daten = pd.read_csv("/mnt/data/INSCYD-Testdaten_der_Athleten.csv")
daten["FFM"] = daten["Gewicht (kg)"] * (1 - daten["Körperfett (%)"] / 100)
daten["Geschlecht_code"] = daten["Geschlecht"].str.lower().map({"mann": 0, "frau": 1})

X = daten[["FFM", "Sprintdauer (s)", "Watt Durchschnitt", "Watt Peak", "Geschlecht_code"]]
y = daten["VLamax INSCYD (mmol/l/s)"]
modell = LinearRegression().fit(X, y)

# ---- Vorhersage ----
if st.button("VLamax berechnen"):
    prediction = modell.predict([[ffm, dauer, watt_avg, watt_peak, geschlecht_code]])[0]
    st.success(f"Geschätzte VLamax: {prediction:.3f} mmol/l/s")

    # Batterie-Anzeige als HTML + CSS
    components.html(f"""
    <div style='display: flex; flex-direction: column; align-items: center;'>
        <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>VLamax</div>
        <div style='width: 60px; height: 200px; border: 2px solid #333; border-radius: 6px; position: relative; background: #e0e0e0;'>
            <div style='width: 100%; background: #4CAF50; position: absolute; bottom: 0; height: {min(max(prediction, 0), 1) * 100:.1f}%'></div>
        </div>
        <div style='margin-top: 8px; font-weight: bold;'>{prediction:.2f} mmol/l/s</div>
    </div>
    """, height=280)

# ---- Modell speichern ----
joblib.dump(modell, "vlamax_model.joblib")