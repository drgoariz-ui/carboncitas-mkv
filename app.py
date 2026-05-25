import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ============================================
# CARGAR MODELO
# ============================================

model = joblib.load("modelo.pkl")
encoder = joblib.load("encoder.pkl")

# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================

st.set_page_config(
    page_title="Carboncitas MKV",
    page_icon="⚙️",
    layout="wide"
)

# ============================================
# CSS / DISEÑO MODERNO
# ============================================

st.markdown("""
<style>

/* ============================================
FONDO GENERAL
============================================ */

.stApp {

    background:
    linear-gradient(
        135deg,
        #0F172A 0%,
        #111827 35%,
        #1E1B4B 100%
    );

    color: white;
}

/* ============================================
TÍTULOS
============================================ */

h1, h2, h3, h4 {

    color: #F8FAFC;

    font-family: 'Segoe UI', sans-serif;
}

/* ============================================
TEXTO
============================================ */

p, label {

    color: #CBD5E1;

    font-size: 16px;
}

/* ============================================
TARJETAS DE MÉTRICAS
============================================ */

[data-testid="metric-container"] {

    background: rgba(255,255,255,0.08);

    border: 1px solid rgba(255,255,255,0.15);

    backdrop-filter: blur(12px);

    padding: 20px;

    border-radius: 22px;

    box-shadow:
        0px 8px 25px rgba(0,0,0,0.35);

    transition: 0.3s;
}

/* Hover tarjetas */

[data-testid="metric-container"]:hover {

    transform: translateY(-5px);

    border: 1px solid #38BDF8;

    box-shadow:
        0px 0px 20px rgba(56,189,248,0.35);
}

/* ============================================
SIDEBAR
============================================ */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #111827,
        #1E293B
    );

    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ============================================
SLIDER
============================================ */

.stSlider > div > div {

    color: #38BDF8;
}

/* ============================================
BOTONES
============================================ */

.stButton button {

    background:
    linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    );

    color: white;

    border-radius: 12px;

    border: none;

    padding: 12px 24px;

    font-weight: bold;
}

/* ============================================
GRÁFICAS
============================================ */

.js-plotly-plot {

    border-radius: 20px;

    overflow: hidden;

    box-shadow:
        0px 8px 30px rgba(0,0,0,0.4);
}

/* ============================================
SEPARADORES
============================================ */

hr {

    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER CON LOGOS
# ============================================

col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.image("unam.png", width=130)

with col2:

    st.markdown("""
    <div style='text-align: center;'>

    <h1 style='font-size: 52px;'>
    ⚙️ Carboncitas MKV
    </h1>

    <h3>
    Universidad Nacional Autónoma de México
    </h3>

    <h4>
    Facultad de Ingeniería
    </h4>

    <p>
    Predicción inteligente de propiedades mecánicas del acero
    mediante Machine Learning
    </p>

    </div>
    """, unsafe_allow_html=True)

with col3:
    st.image("fiunam.png", width=130)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("⚙️ Parámetros")

carbono = st.sidebar.slider(
    "% de Carbono",
    0.0,
    1.0,
    0.40,
    0.01
)

tratamiento = st.sidebar.selectbox(
    "Tratamiento térmico",
    encoder.classes_
)

# ============================================
# PREPARAR DATOS
# ============================================

tratamiento_cod = encoder.transform([tratamiento])[0]

X = pd.DataFrame({
    "C_pct": [carbono],
    "Condition_encoded": [tratamiento_cod]
})

# ============================================
# PREDICCIÓN
# ============================================

pred = model.predict(X)[0]

uts = pred[0]
ys = pred[1]
hardness = pred[2]
elong = pred[3]

# ============================================
# RESULTADOS
# ============================================

st.header("📊 Resultados predichos")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "UTS (MPa)",
    f"{uts:.1f}"
)

c2.metric(
    "YS (MPa)",
    f"{ys:.1f}"
)

c3.metric(
    "Dureza (HB)",
    f"{hardness:.1f}"
)

c4.metric(
    "Elongación (%)",
    f"{elong:.1f}"
)

# ============================================
# GAUGES
# ============================================

st.header("📈 Indicadores visuales")

g1, g2 = st.columns(2)

with g1:

    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=uts,
        title={'text': "UTS (MPa)"},
        gauge={
            'axis': {'range': [0, 1400]},
            'bar': {'color': "#38BDF8"}
        }
    ))

    fig1.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with g2:

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hardness,
        title={'text': "Dureza (HB)"},
        gauge={
            'axis': {'range': [0, 400]},
            'bar': {'color': "#A855F7"}
        }
    ))

    fig2.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ============================================
# GRÁFICA DE BARRAS
# ============================================

st.header("📉 Comparación de propiedades")

fig_bar = px.bar(
    x=["UTS", "YS", "Hardness", "Elongation"],
    y=[uts, ys, hardness, elong],
    color=[
        "UTS",
        "YS",
        "Hardness",
        "Elongation"
    ],
    text=[
        f"{uts:.1f}",
        f"{ys:.1f}",
        f"{hardness:.1f}",
        f"{elong:.1f}"
    ]
)

fig_bar.update_layout(
    height=500,
    showlegend=False,
    template="plotly_dark"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

# ============================================
# TENDENCIA INTERACTIVA
# ============================================

st.header("📈 Tendencia de propiedades mecánicas")

propiedad = st.selectbox(
    "Selecciona una propiedad",
    [
        "UTS (MPa)",
        "YS (MPa)",
        "Hardness (HB)",
        "Elongation (%)"
    ]
)

carbonos = np.linspace(0, 1, 50)

valores = []

for c in carbonos:

    X_temp = pd.DataFrame({
        "C_pct": [c],
        "Condition_encoded": [tratamiento_cod]
    })

    p = model.predict(X_temp)[0]

    if propiedad == "UTS (MPa)":
        valores.append(p[0])
        valor_actual = uts

    elif propiedad == "YS (MPa)":
        valores.append(p[1])
        valor_actual = ys

    elif propiedad == "Hardness (HB)":
        valores.append(p[2])
        valor_actual = hardness

    elif propiedad == "Elongation (%)":
        valores.append(p[3])
        valor_actual = elong

# ============================================
# CREAR GRÁFICA
# ============================================

fig_prop = go.Figure()

fig_prop.add_trace(go.Scatter(
    x=carbonos,
    y=valores,
    mode='lines',
    line=dict(width=5),
    name=propiedad
))

fig_prop.add_trace(go.Scatter(
    x=[carbono],
    y=[valor_actual],
    mode='markers',
    marker=dict(size=14),
    name='Punto actual'
))

fig_prop.update_layout(

    template="plotly_dark",

    height=600,

    title=f"{propiedad} vs % de Carbono",

    xaxis_title="% de Carbono",

    yaxis_title=propiedad,

    hovermode="x unified"
)

st.plotly_chart(
    fig_prop,
    use_container_width=True
)

# ============================================
# INTERPRETACIÓN IA
# ============================================

st.header("🧠 Interpretación automática")

if carbono > 0.6:

    st.success(
        "El material presenta alta resistencia y dureza, "
        "pero menor ductilidad debido al incremento de carbono."
    )

elif carbono < 0.2:

    st.info(
        "El acero presenta alta ductilidad y menor resistencia, "
        "característico de aceros bajos en carbono."
    )

else:

    st.warning(
        "El acero presenta propiedades mecánicas balanceadas "
        "entre resistencia y ductilidad."
    )

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown("""
Proyecto de Ingeniería de Materiales elaborado por:  

Corona Rodríguez Marjan Fátima  
Gutiérrez Bueno Valeria  
Serrano López Kenny Gabriela  

Desarrollado con Python, Streamlit y Machine Learning
""")

