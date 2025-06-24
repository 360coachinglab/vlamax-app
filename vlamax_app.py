# VLamax Predictor Web-App mit erweitertem Modell und CSV-Update-Funktion
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import streamlit as st
import streamlit.components.v1 as components
import os

CSV_PATH = "vlamax_testdaten.csv"

# ---- Web App Layout ----
st.image("https://raw.githubusercontent.com/360coachinglab/vlamax-app/main/360coachinglab%20gross.png", width=300)
st.title("VLamax Auswertung")
st.write("Erweiterte Schätzung basierend auf Geschlecht, Körperkomposition und Sprintleistung")

# ---- Eingabe neue Testdaten zur Ergänzung der Datenbank ----
st.header("📝 Testdatenbank erweitern")
with st.expander("Neue Testdaten hinzufügen"):
    col1, col2 = st.columns(2)
    with col1:
        neuer_name = st.text_input("Athletenname")
        neues_geschlecht = st.selectbox("Geschlecht", ["Mann", "Frau"], key="neu")
        neues_gewicht = st.number_input("Gewicht (kg)", value=70.0, step=0.1, key="g1")
        neues_fett = st.number_input("Körperfett (%)", value=15.0, step=0.1, key="f1")
        neue_groesse = st.number_input("Grösse (cm)", value=175, step=1, key="g2")
    with col2:
        neues_alter = st.number_input("Alter", value=25, step=1, key="a1")
        neue_dauer = st.number_input("Sprintdauer (s)", value=20, step=1, key="s1")
        neue_watt_avg = st.number_input("Watt Durchschnitt", value=650, step=10, key="w1")
        neue_watt_peak = st.number_input("Watt Peak", value=900, step=10, key="w2")
        neue_vlamax = st.number_input("VLamax (mmol/l/s)", value=0.45, step=0.01, key="v1")

    if st.button("Testdaten hinzufügen"):
        neue_daten = pd.DataFrame({
            "Athlet": [neuer_name],
            "Geschlecht": [neues_geschlecht],
            "Gewicht (kg)": [neues_gewicht],
            "Körperfett (%)": [neues_fett],
            "Grösse (cm)": [neue_groesse],
            "Alter": [neues_alter],
            "Sprintdauer (s)": [neue_dauer],
            "Watt Durchschnitt": [neue_watt_avg],
            "Watt Peak": [neue_watt_peak],
            "VLamax INSCYD (mmol/l/s)": [neue_vlamax]
        })
        if os.path.exists(CSV_PATH):
            bestehend = pd.read_csv(CSV_PATH)
            pd.concat([bestehend, neue_daten], ignore_index=True).to_csv(CSV_PATH, index=False)
        else:
            neue_daten.to_csv(CSV_PATH, index=False)
        st.success("Testdaten erfolgreich hinzugefügt!")

# ---- Eingabe zur VLamax-Schätzung ----
st.header("📊 VLamax berechnen")
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

# ---- Modelltraining mit Datenprüfung ----
if os.path.exists(CSV_PATH):
    daten = pd.read_csv(CSV_PATH)
    if len(daten) < 2:
        st.warning("Nicht genügend Daten zum Trainieren des Modells. Bitte ergänze mehr Testdaten.")
    else:
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

        joblib.dump(modell, "vlamax_model.joblib")
else:
    st.warning("CSV-Datei nicht gefunden. Bitte lade zuerst Testdaten hoch.")