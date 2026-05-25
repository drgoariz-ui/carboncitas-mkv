import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# ============================================
# CARGAR DATOS
# ============================================

df = pd.read_csv(
    "data/stainless.csv",
    encoding="latin1"
)

# ============================================
# SELECCIONAR COLUMNAS IMPORTANTES
# ============================================

columnas = [
    "Conditions",
    "UTS (MPa)",
    "YS (MPa)",
    "Elongation (%)",
    "Hardness (HB)",
    "C (Max)",
    "Cr (Max)",
    "Ni (Max)",
    "Mn (Max)"
]

df = df[columnas]

# ============================================
# CLASIFICAR TRATAMIENTOS
# ============================================

def clasificar_tratamiento(cond):

    cond = str(cond).lower()

    if "annealed" in cond:
        return "Annealed"

    elif "solution" in cond:
        return "Solution Treated"

    elif "cold" in cond:
        return "Cold Worked"

    elif "hot" in cond:
        return "Hot Rolled"

    elif "quenched" in cond:
        return "Quenched"

    else:
        return "Other"


df["Condition_simple"] = df["Conditions"].apply(
    clasificar_tratamiento
)

# ============================================
# LIMPIAR DUREZA
# ============================================

df["Hardness (HB)"] = (
    df["Hardness (HB)"]
    .astype(str)
    .str.extract(r'(\d+)')[0]
)

df["Hardness (HB)"] = pd.to_numeric(
    df["Hardness (HB)"],
    errors="coerce"
)

# ============================================
# CONVERTIR A NUMÉRICO
# ============================================

columnas_numericas = [
    "UTS (MPa)",
    "YS (MPa)",
    "Elongation (%)",
    "Hardness (HB)",
    "C (Max)",
    "Cr (Max)",
    "Ni (Max)",
    "Mn (Max)"
]

for col in columnas_numericas:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ============================================
# ELIMINAR NaNs IMPORTANTES
# ============================================

df = df.dropna(subset=[
    "UTS (MPa)",
    "YS (MPa)",
    "Elongation (%)",
    "Hardness (HB)",
    "Cr (Max)",
    "Ni (Max)",
    "Condition_simple"
])

# ============================================
# CODIFICAR TRATAMIENTOS
# ============================================

encoder = LabelEncoder()

df["Condition_encoded"] = encoder.fit_transform(
    df["Condition_simple"]
)

# ============================================
# VARIABLES DE ENTRADA
# ============================================

X = df[[
    "Cr (Max)",
    "Ni (Max)",
    "C (Max)",
    "Mn (Max)",
    "Condition_encoded"
]]

# ============================================
# VARIABLES OBJETIVO
# ============================================

y = df[[
    "UTS (MPa)",
    "YS (MPa)",
    "Hardness (HB)",
    "Elongation (%)"
]]

# ============================================
# MODELO IA
# ============================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Entrenar
model.fit(X, y)

# ============================================
# GUARDAR MODELO
# ============================================

joblib.dump(model, "modelo_stainless.pkl")
joblib.dump(encoder, "encoder_stainless.pkl")

print("Modelo Stainless entrenado correctamente")