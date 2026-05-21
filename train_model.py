import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# =========================
# CARGAR DATOS
# =========================

df = pd.read_csv(
    "data/steel.csv",
    encoding="latin1"
)

# =========================
# LIMPIEZA DE DATOS
# =========================

# Carbono promedio
df["C_pct"] = (
    df["C (Min)"] + df["C (Max)"]
) / 2

# ============================================
# CLASIFICAR TRATAMIENTOS
# ============================================

def clasificar_tratamiento(cond):

    cond = str(cond).lower()

    if "annealed" in cond:
        return "Annealed"

    elif "normalized" in cond:
        return "Normalized"

    elif "hot rolled" in cond:
        return "Hot Rolled"

    elif "cold drawn" in cond:
        return "Cold Drawn"

    elif "quenched" in cond:
        return "Quenched"

    elif "spheroidized" in cond:
        return "Spheroidized"

    elif "oi" in cond:
        return "Other"

    else:
        return "Other"


df["Condition_simple"] = df["Conditions"].apply(
    clasificar_tratamiento
)

# Limpiar columna de dureza
# Ejemplo: "269 HRB" -> 269

df["Hardness (HB)"] = (
    df["Hardness (HB)"]
    .astype(str)
    .str.extract(r'(\d+)')[0]
)

df["Hardness (HB)"] = pd.to_numeric(
    df["Hardness (HB)"],
    errors="coerce"
)

# Convertir demás columnas a número

columnas_numericas = [
    "UTS (MPa)",
    "YS (MPa)",
    "Elongation (%)",
    "C_pct"
]

for col in columnas_numericas:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Eliminar filas con NaNs importantes

df = df.dropna(subset=[
    "C_pct",
    "UTS (MPa)",
    "YS (MPa)",
    "Hardness (HB)",
    "Elongation (%)",
    "Condition_simple"
])

# =========================
# CODIFICAR TRATAMIENTOS
# =========================

encoder = LabelEncoder()

df["Condition_encoded"] = encoder.fit_transform(
    df["Condition_simple"]
)

# =========================
# VARIABLES DE ENTRADA
# =========================

X = df[[
    "C_pct",
    "Condition_encoded"
]]

# =========================
# VARIABLES OBJETIVO
# =========================

y = df[[
    "UTS (MPa)",
    "YS (MPa)",
    "Hardness (HB)",
    "Elongation (%)"
]]

# =========================
# MODELO IA
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Entrenar modelo
model.fit(X, y)

# =========================
# GUARDAR MODELO
# =========================

joblib.dump(model, "modelo.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Modelo entrenado correctamente")
