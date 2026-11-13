import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="Proyecto Integrador U4",
    page_icon="🌱",
    layout="wide"
)

# ==================================================
# FUNCION 1: GENERAR DATOS AGRICOLAS
# ==================================================

@st.cache_data
def generar_datos():
    con = duckdb.connect(database=":memory:")
    con.execute("SELECT setseed(0.42);")

    con.execute("""
    CREATE TABLE parcelas_agricolas AS

    WITH datos_base AS MATERIALIZED (
        SELECT
            parcela_id,

            ROUND(
                100 + random() * 200,
                2
            ) AS volumen_riego,

            ROUND(
                50 + random() * 100,
                2
            ) AS fertilizante_kg,

            CASE
                WHEN random() < 0.50
                THEN 'Tratamiento_A'
                ELSE 'Tratamiento_B'
            END AS tipo_tratamiento,

            (random() * 1.6 - 0.8)
                AS ruido_aleatorio

        FROM generate_series(1, 100)
            AS t(parcela_id)
    )

    SELECT
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,

        ROUND(
            5.0
            + 0.035 * volumen_riego
            + 0.020 * fertilizante_kg
            + CASE
                WHEN tipo_tratamiento = 'Tratamiento_B'
                THEN 1.8
                ELSE 0.0
              END
            + ruido_aleatorio,
            2
        ) AS rendimiento_ton

    FROM datos_base;
    """)

    df = con.execute("""
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            rendimiento_ton
        FROM parcelas_agricolas
        ORDER BY parcela_id
    """).df()

    con.close()
    return df

# ==================================================
# FUNCION 2: ANALISIS DE REGRESION
# ==================================================

def calcular_modelo(df):
    variables_numericas = [
        "volumen_riego",
        "fertilizante_kg",
        "rendimiento_ton"
    ]

    matriz_correlacion = df[variables_numericas].corr(method="pearson")

    X = df[["volumen_riego", "fertilizante_kg"]]
    y = df["rendimiento_ton"]

    X_const = sm.add_constant(X)
    modelo = sm.OLS(y, X_const).fit()
    predicciones = modelo.predict(X_const)
    rmse = np.sqrt(mean_squared_error(y, predicciones))

    return matriz_correlacion, modelo, rmse

# ==================================================
# FUNCION 3: PRUEBA DE HIPOTESIS
# ==================================================

def prueba_tratamientos(df):
    tratamiento_a = df.loc[
        df["tipo_tratamiento"] == "Tratamiento_A",
        "rendimiento_ton"
    ]

    tratamiento_b = df.loc[
        df["tipo_tratamiento"] == "Tratamiento_B",
        "rendimiento_ton"
    ]

    if len(tratamiento_a) == 0 or len(tratamiento_b) == 0:
        return None

    media_a = tratamiento_a.mean()
    media_b = tratamiento_b.mean()
    diferencia = media_b - media_a

    resultado_t = stats.ttest_ind(
        tratamiento_b,
        tratamiento_a,
        equal_var=False,
        alternative="greater"
    )

    return {
        "media_a": media_a,
        "media_b": media_b,
        "diferencia": diferencia,
        "t": resultado_t.statistic,
        "p": resultado_t.pvalue,
        "n_a": len(tratamiento_a),
        "n_b": len(tratamiento_b)
    }

# ==================================================
# CARGA DE DATOS
# ==================================================

df = generar_datos()

# ==================================================
# BARRA LATERAL
# ==================================================

st.sidebar.title("⚙️ Filtros")

opcion_tratamiento = st.sidebar.selectbox(
    "Selecciona el tratamiento",
    ["Todos", "Tratamiento_A", "Tratamiento_B"]
)

if opcion_tratamiento == "Todos":
    df_filtrado = df.copy()
else:
    df_filtrado = df[
        df["tipo_tratamiento"] == opcion_tratamiento
    ].copy()

# Calcular análisis con datos filtrados
matriz_correlacion, modelo_ols, rmse = calcular_modelo(df_filtrado)

# ==================================================
# TITULO
# ==================================================

st.title("🌱 Proyecto Integrador U4")
st.subheader("Optimización inferencial del rendimiento agrícola y eficiencia de riego")

st.markdown("""
Este dashboard analiza el efecto del **volumen de riego**, el **fertilizante**
y el **tipo de tratamiento agrícola** sobre el **rendimiento del cultivo**.
""")

# ==================================================
# INDICADORES PRINCIPALES
# ==================================================

# INDICADORES PRINCIPALES SOLICITADOS

resultado_global = prueba_tratamientos(df)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rendimiento promedio global",
        f"{df['rendimiento_ton'].mean():.2f} ton/ha"
    )

with col2:
    st.metric(
        "R² ajustado del modelo",
        f"{modelo_ols.rsquared_adj:.4f}"
    )

with col3:
    st.metric(
        "Valor p de la prueba inferencial",
        f"{resultado_global['p']:.4e}"
    )


st.markdown("---")

# ==================================================
# RESULTADO INFERENCIAL
# ==================================================

st.subheader("📌 Comparación inferencial entre tratamientos")

resultado_prueba = prueba_tratamientos(df)

if resultado_prueba is not None:
    st.write(f"**Media Tratamiento A:** {resultado_prueba['media_a']:.4f} ton/ha")
    st.write(f"**Media Tratamiento B:** {resultado_prueba['media_b']:.4f} ton/ha")
    st.write(f"**Diferencia B - A:** {resultado_prueba['diferencia']:.4f} ton/ha")
    st.write(f"**Estadístico t:** {resultado_prueba['t']:.4f}")
    st.write(f"**Valor p:** {resultado_prueba['p']:.8e}")

    if resultado_prueba["p"] < 0.05:
        st.success(
            "Se rechaza H0: existe evidencia estadística de que el tratamiento B aumenta el rendimiento."
        )
    else:
        st.warning(
            "No se rechaza H0: no existe evidencia suficiente para afirmar que el tratamiento B mejora el rendimiento."
        )

st.markdown("---")

# ==================================================
# TABLA DE DATOS
# ==================================================

st.subheader("📋 Datos agrícolas filtrados")
st.dataframe(df_filtrado, use_container_width=True)

st.markdown("---")

# ==================================================
# GRAFICAS 2x2
# ==================================================

st.subheader("📊 Visualizaciones del proyecto")

sns.set_theme(style="whitegrid")

# -------- GRAFICA 1: MATRIZ DE CORRELACION --------
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.heatmap(
    matriz_correlacion,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax1
)
ax1.set_title("Matriz de correlación")

# -------- GRAFICA 2: RIEGO VS RENDIMIENTO --------
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.regplot(
    data=df_filtrado,
    x="volumen_riego",
    y="rendimiento_ton",
    ci=95,
    scatter_kws={"alpha": 0.65, "s": 55},
    line_kws={"color": "red", "linewidth": 2},
    ax=ax2
)
ax2.set_title("Riego vs rendimiento")
ax2.set_xlabel("Volumen de riego (m³/ha)")
ax2.set_ylabel("Rendimiento (ton/ha)")

# -------- GRAFICA 3: BOXPLOT DE TRATAMIENTOS --------
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.boxplot(
    data=df_filtrado,
    x="tipo_tratamiento",
    y="rendimiento_ton",
    showmeans=True,
    ax=ax3
)
ax3.set_title("Rendimiento por tratamiento")
ax3.set_xlabel("Tratamiento")
ax3.set_ylabel("Rendimiento (ton/ha)")

# -------- GRAFICA 4: HISTOGRAMA DE RESIDUOS --------
fig4, ax4 = plt.subplots(figsize=(6, 4))
residuos = modelo_ols.resid
sns.histplot(
    residuos,
    bins=15,
    kde=True,
    ax=ax4
)
ax4.axvline(
    x=0,
    color="red",
    linestyle="--",
    label="Residuo cero"
)
ax4.set_title("Histograma de residuos")
ax4.set_xlabel("Residuo")
ax4.set_ylabel("Frecuencia")
ax4.legend()

# Mostrar en distribución 2x2
c1, c2 = st.columns(2)
with c1:
    st.pyplot(fig1)
with c2:
    st.pyplot(fig2)

c3, c4 = st.columns(2)
with c3:
    st.pyplot(fig3)
with c4:
    st.pyplot(fig4)

st.markdown("---")

# ==================================================
# RESUMEN DEL MODELO
# ==================================================

st.subheader("🧮 Resumen del modelo OLS")

st.write("**Coeficientes estimados:**")
coef_df = pd.DataFrame({
    "Variable": modelo_ols.params.index,
    "Coeficiente": modelo_ols.params.values
})
st.dataframe(coef_df, use_container_width=True)

st.write(f"**R²:** {modelo_ols.rsquared:.4f}")
st.write(f"**R² ajustado:** {modelo_ols.rsquared_adj:.4f}")
st.write(f"**RMSE:** {rmse:.4f} ton/ha")
st.write(f"**Valor p global del modelo:** {modelo_ols.f_pvalue:.8e}")

with st.expander("Ver resumen estadístico completo del modelo"):
    st.text(modelo_ols.summary().as_text())
