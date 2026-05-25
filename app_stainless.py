import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

st.set_page_config(
    page_title="Carboncitas inoxidables",
    layout="wide",
    page_icon="⚙️"
)

# ============================================
# CARGAR MODELO
# ============================================

model = joblib.load("modelo_stainless.pkl")
encoder = joblib.load("encoder_stainless.pkl")

# ============================================
# ESTILO VISUAL PREMIUM
# ============================================

st.markdown("""
<style>

/* Fondo general */

.stApp {
    background: linear-gradient(
        135deg,
        #020617 0%,
        #0f172a 40%,
        #111827 100%
    );
    color: white;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a,
        #111827
    );
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Títulos */

h1 {
    color: #67e8f9;
    font-size: 55px !important;
    text-shadow: 0px 0px 15px rgba(103,232,249,0.5);
}

h2 {
    color: #93c5fd;
    text-shadow: 0px 0px 10px rgba(147,197,253,0.3);
}

/* Tarjetas métricas */

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(59,130,246,0.15);
    text-align: center;
    transition: 0.3s;
}

.metric-card:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 25px rgba(103,232,249,0.4);
}

/* Texto de métricas */

.metric-title {
    color: #93c5fd;
    font-size: 20px;
    font-weight: bold;
}

.metric-value {
    color: white;
    font-size: 40px;
    font-weight: bold;
}

/* Footer */

.footer {
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Separadores */

hr {
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

col_logo1, col_title, col_logo2 = st.columns([1,4,1])

with col_logo1:
    st.image("unam.png", width=120)

with col_title:

    st.markdown(
        """
        <h1 style='text-align:center;'>
        ⚙️ Carboncitas inoxidables
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style='text-align:center; color:#cbd5e1;'>
        Predicción inteligente de propiedades mecánicas en aceros inoxidables
        </h3>
        """,
        unsafe_allow_html=True
    )

with col_logo2:
    st.image("fiunam.png", width=120)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("⚙️ Parámetros")

cr = st.sidebar.slider(
    "% Cr (Cromo)",
    10.0,
    30.0,
    18.0
)

ni = st.sidebar.slider(
    "% Ni (Níquel)",
    0.0,
    25.0,
    8.0
)

c = st.sidebar.slider(
    "% Carbono",
    0.0,
    1.0,
    0.08
)

mn = st.sidebar.slider(
    "% Manganeso",
    0.0,
    5.0,
    1.0
)

tratamiento = st.sidebar.selectbox(
    "Tratamiento térmico",
    encoder.classes_
)

# ============================================
# PREPARAR DATOS
# ============================================

tratamiento_encoded = encoder.transform(
    [tratamiento]
)[0]

entrada = pd.DataFrame([[
    cr,
    ni,
    c,
    mn,
    tratamiento_encoded
]], columns=[
    "Cr (Max)",
    "Ni (Max)",
    "C (Max)",
    "Mn (Max)",
    "Condition_encoded"
])

# ============================================
# PREDICCIÓN
# ============================================

pred = model.predict(entrada)

uts = pred[0][0]
ys = pred[0][1]
hardness = pred[0][2]
elong = pred[0][3]

# ============================================
# RESULTADOS PREMIUM
# ============================================

st.markdown("## 📊 Resultados predichos")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⚙️ UTS</div>
        <div class="metric-value">{uts:.1f}</div>
        <div>MPa</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🛡️ YS</div>
        <div class="metric-value">{ys:.1f}</div>
        <div>MPa</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🔩 Dureza</div>
        <div class="metric-value">{hardness:.1f}</div>
        <div>HB</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🧪 Elongación</div>
        <div class="metric-value">{elong:.1f}</div>
        <div>%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# DESEMPEÑO MECÁNICO RELATIVO
# ============================================

st.markdown("## 📈 Desempeño mecánico relativo")

uts_norm = uts / 1200
ys_norm = ys / 1200
hardness_norm = hardness / 400
elong_norm = elong / 100

perfil_df = pd.DataFrame({
    "Propiedad": [
        "UTS",
        "YS",
        "Dureza",
        "Elongación"
    ],
    "Nivel relativo": [
        uts_norm,
        ys_norm,
        hardness_norm,
        elong_norm
    ]
})

fig_perf = px.bar(
    perfil_df,
    x="Nivel relativo",
    y="Propiedad",
    orientation="h",
    text="Nivel relativo",
    template="plotly_dark",
    text_auto=".2f"
)

fig_perf.update_traces(
    marker_color=[
        "#38bdf8",
        "#60a5fa",
        "#818cf8",
        "#22d3ee"
    ]
)

fig_perf.update_layout(
    height=500,
    xaxis_title="Nivel relativo",
    yaxis_title="",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="white",
        size=16
    )
)

st.plotly_chart(
    fig_perf,
    use_container_width=True
)

# ============================================
# FUNCIÓN CACHEADA PARA SUPERFICIE 3D
# ============================================

@st.cache_data
def generar_superficie(
    propiedad,
    c,
    mn,
    tratamiento_encoded
):

    # Menos puntos = más rápido
    cr_vals = np.linspace(10, 30, 15)
    ni_vals = np.linspace(0, 25, 15)

    CR, NI = np.meshgrid(
        cr_vals,
        ni_vals
    )

    Z = np.zeros_like(CR)

    for i in range(CR.shape[0]):
        for j in range(CR.shape[1]):

            entrada_3d = pd.DataFrame([[
                CR[i,j],
                NI[i,j],
                c,
                mn,
                tratamiento_encoded
            ]], columns=[
                "Cr (Max)",
                "Ni (Max)",
                "C (Max)",
                "Mn (Max)",
                "Condition_encoded"
            ])

            pred_3d = model.predict(
                entrada_3d
            )

            if propiedad == "UTS":
                Z[i,j] = pred_3d[0][0]

            elif propiedad == "YS":
                Z[i,j] = pred_3d[0][1]

            elif propiedad == "Dureza":
                Z[i,j] = pred_3d[0][2]

            elif propiedad == "Elongación":
                Z[i,j] = pred_3d[0][3]

    return CR, NI, Z

# ============================================
# EXPLORACIÓN 3D
# ============================================

st.markdown("## 🌐 Exploración 3D del material")

propiedad_3d = st.selectbox(
    "Selecciona propiedad a visualizar",
    [
        "UTS",
        "YS",
        "Dureza",
        "Elongación"
    ]
)

CR, NI, Z = generar_superficie(
    propiedad_3d,
    c,
    mn,
    tratamiento_encoded
)

fig_3d = go.Figure(data=[
    go.Surface(
        x=CR,
        y=NI,
        z=Z,
        colorscale="Viridis"
    )
])

fig_3d.update_layout(
    template="plotly_dark",
    height=700,
    title=f"Superficie 3D de {propiedad_3d}",
    scene=dict(
        xaxis_title="% Cr",
        yaxis_title="% Ni",
        zaxis_title=propiedad_3d,
        bgcolor="rgba(0,0,0,0)"
    ),
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(
    fig_3d,
    use_container_width=True
)

# ============================================
# INTERPRETACIÓN AUTOMÁTICA
# ============================================

st.markdown("## 🧠 Interpretación inteligente")

if uts > 900:
    st.success(
        "⚙️ El material presenta muy alta resistencia mecánica."
    )

elif uts > 700:
    st.info(
        "⚙️ El material presenta resistencia mecánica moderada-alta."
    )

else:
    st.warning(
        "⚙️ El material presenta resistencia mecánica relativamente baja."
    )

if elong > 30:
    st.success(
        "🧪 El material presenta alta ductilidad."
    )

elif elong > 15:
    st.info(
        "🧪 El material presenta ductilidad moderada."
    )

else:
    st.warning(
        "🧪 El material presenta baja ductilidad."
    )

if hardness > 250:
    st.success(
        "🔩 El material presenta alta dureza superficial."
    )

elif hardness > 180:
    st.info(
        "🔩 El material presenta dureza moderada."
    )

else:
    st.warning(
        "🔩 El material presenta dureza relativamente baja."
    )

# ============================================
# FOOTER PREMIUM
# ============================================

st.markdown("---")

st.markdown("""
<div class="footer">

<h3>🏛️ Proyecto de Ingeniería de Materiales — UNAM FI</h3>

<h4>💻 Desarrollado con:</h4>

<ul>
<li>Python</li>
<li>Streamlit</li>
<li>Machine Learning</li>
<li>Ciencia de Datos</li>
</ul>

<h4>👨‍🔬 Elaborado por:</h4>

<ul>
<li>Corona Rodríguez Marjan Fátima</li>
<li>Gutiérrez Bueno Valeria</li>
<li>Serrano López Kenny Gabriela</li>
</ul>

</div>
""", unsafe_allow_html=True)