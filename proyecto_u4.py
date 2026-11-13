
import duckdb
import pandas as pd
import numpy as np

# ==================================================
# PROYECTO INTEGRADOR UNIDAD 4
# OPTIMIZACION DEL RENDIMIENTO AGRICOLA
#
# FASE 1: SIMULACION DE TELEMETRIA CON DUCKDB
# ==================================================

# Crear una base de datos en memoria
con = duckdb.connect(database=":memory:")

# Fijar semilla determinista
con.execute("SELECT setseed(0.42);")

# ==================================================
# PASO 1: GENERAR 100 PARCELAS AGRICOLAS
# ==================================================

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

# ==================================================
# PASO 2: EXTRAER DATOS MEDIANTE SQL
# ==================================================

df_parcelas = con.execute("""
    SELECT
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        rendimiento_ton
    FROM parcelas_agricolas
    ORDER BY parcela_id
""").df()

# ==================================================
# PASO 3: VALIDAR RESULTADOS
# ==================================================

print("=" * 65)
print("PROYECTO INTEGRADOR UNIDAD 4")
print("FASE 1: SIMULACION DE DATOS AGRICOLAS")
print("=" * 65)

print("\nPRIMEROS 10 REGISTROS:")
print(df_parcelas.head(10).to_string(index=False))

print("\nTOTAL DE PARCELAS:")
print(len(df_parcelas))

print("\nCOLUMNAS DISPONIBLES:")
print(df_parcelas.columns.tolist())

print("\nDISTRIBUCION DE TRATAMIENTOS:")
print(
    df_parcelas["tipo_tratamiento"].value_counts()
)

print("\nRESUMEN ESTADISTICO:")
print(
    df_parcelas[
        [
            "volumen_riego",
            "fertilizante_kg",
            "rendimiento_ton"
        ]
    ].describe().round(2)
)

print("\nVALORES NULOS POR COLUMNA:")
print(df_parcelas.isnull().sum())

print("\n" + "=" * 65)
print("FASE 1 COMPLETADA")
print("=" * 65)

con.close()



# ==================================================
# FASE 2: CORRELACION Y REGRESION LINEAL MULTIPLE
# ==================================================

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

print("\n" + "=" * 65)
print("FASE 2: CORRELACION Y REGRESION LINEAL")
print("=" * 65)

# PASO 2.1: MATRIZ DE CORRELACION DE PEARSON

variables_numericas = [
    "volumen_riego",
    "fertilizante_kg",
    "rendimiento_ton"
]

matriz_correlacion = df_parcelas[
    variables_numericas
].corr(method="pearson")

print("\nMATRIZ DE CORRELACION DE PEARSON:")
print(matriz_correlacion.round(4).to_string())

print("\nCORRELACION CON EL RENDIMIENTO:")
print(
    matriz_correlacion["rendimiento_ton"]
    .drop("rendimiento_ton")
    .round(4)
    .to_string()
)

# Identificar variable mas correlacionada
correlaciones = (
    matriz_correlacion["rendimiento_ton"]
    .drop("rendimiento_ton")
)

variable_dominante = correlaciones.abs().idxmax()

print("\nVARIABLE CON MAYOR CORRELACION ABSOLUTA:")
print(variable_dominante)
print(f"Coeficiente r: {correlaciones[variable_dominante]:.4f}")

# PASO 2.2: MODELO DE REGRESION OLS

X = df_parcelas[
    ["volumen_riego", "fertilizante_kg"]
]

Y = df_parcelas["rendimiento_ton"]

X_const = sm.add_constant(X)

modelo_ols = sm.OLS(Y, X_const).fit()

print("\n" + "=" * 65)
print("MODELO DE REGRESION LINEAL MULTIPLE OLS")
print("=" * 65)

print(modelo_ols.summary())

# PASO 2.3: COEFICIENTES Y METRICAS

predicciones_ols = modelo_ols.predict(X_const)

rmse_ols = np.sqrt(
    mean_squared_error(Y, predicciones_ols)
)

print("\nCOEFICIENTES DEL MODELO:")
for nombre, valor in modelo_ols.params.items():
    print(f"{nombre}: {valor:.6f}")

print("\nMETRICAS DEL MODELO:")
print(f"R2: {modelo_ols.rsquared:.4f}")
print(f"R2 ajustado: {modelo_ols.rsquared_adj:.4f}")
print(f"RMSE: {rmse_ols:.4f} ton/ha")
print(f"Valor p global: {modelo_ols.f_pvalue:.6e}")

print("\nECUACION ESTIMADA:")
print(
    f"Rendimiento = {modelo_ols.params['const']:.4f}"
    f" + ({modelo_ols.params['volumen_riego']:.4f} * Riego)"
    f" + ({modelo_ols.params['fertilizante_kg']:.4f} * Fertilizante)"
)

print("\n" + "=" * 65)
print("FASE 2 COMPLETADA")
print("=" * 65)



# ==================================================
# FASE 3: PRUEBA DE HIPOTESIS DE TRATAMIENTOS
# ==================================================

from scipy import stats

print("\n" + "=" * 65)
print("FASE 3: INFERENCIA ESTADISTICA DE TRATAMIENTOS")
print("=" * 65)

# PASO 3.1: SEPARAR LOS DOS TRATAMIENTOS

tratamiento_a = df_parcelas.loc[
    df_parcelas["tipo_tratamiento"] == "Tratamiento_A",
    "rendimiento_ton"
]

tratamiento_b = df_parcelas.loc[
    df_parcelas["tipo_tratamiento"] == "Tratamiento_B",
    "rendimiento_ton"
]

print("\nHIPOTESIS:")
print("H0: mu_B - mu_A <= 0")
print("H1: mu_B - mu_A > 0")
print("Nivel de significancia: 0.05")

print("\nDISTRIBUCION DE MUESTRAS:")
print(f"Tratamiento A: {len(tratamiento_a)} parcelas")
print(f"Tratamiento B: {len(tratamiento_b)} parcelas")

media_a = tratamiento_a.mean()
media_b = tratamiento_b.mean()

diferencia = media_b - media_a

print("\nMEDIAS DE PRODUCCION:")
print(f"Media A: {media_a:.4f} ton/ha")
print(f"Media B: {media_b:.4f} ton/ha")
print(f"Diferencia B - A: {diferencia:.4f} ton/ha")

# PASO 3.2: PRUEBA T DE DOS MUESTRAS
# Welch permite varianzas distintas

resultado_t = stats.ttest_ind(
    tratamiento_b,
    tratamiento_a,
    equal_var=False,
    alternative="greater"
)

t_estadistico = resultado_t.statistic
p_valor = resultado_t.pvalue

print("\nPRUEBA T DE WELCH (UNILATERAL):")
print(f"Estadistico t: {t_estadistico:.4f}")
print(f"Valor p: {p_valor:.8e}")

# PASO 3.3: INTERVALO DE CONFIANZA 95%
# Para la diferencia de medias B - A

n_a = len(tratamiento_a)
n_b = len(tratamiento_b)

var_a = tratamiento_a.var(ddof=1)
var_b = tratamiento_b.var(ddof=1)

se_diferencia = np.sqrt(
    var_a / n_a + var_b / n_b
)

grados_libertad = (
    (var_a / n_a + var_b / n_b) ** 2
    /
    (
        (var_a / n_a) ** 2 / (n_a - 1)
        +
        (var_b / n_b) ** 2 / (n_b - 1)
    )
)

# Intervalo bilateral del 95 %
t_critico = stats.t.ppf(
    0.975,
    grados_libertad
)

ic_inferior = diferencia - t_critico * se_diferencia
ic_superior = diferencia + t_critico * se_diferencia

print("\nINTERVALO DE CONFIANZA BILATERAL DEL 95%:")
print(f"IC: [{ic_inferior:.4f}, {ic_superior:.4f}] ton/ha")

# PASO 3.4: DECISION ESTADISTICA

print("\nDECISION DE INGENIERIA:")

if p_valor < 0.05:
    print("SE RECHAZA LA HIPOTESIS NULA H0")
    print(
        "Existe evidencia estadistica de que "
        "el tratamiento B aumenta el rendimiento."
    )
    print(
        "Se recomienda evaluar su implementacion "
        "considerando tambien los costos."
    )
else:
    print("NO SE RECHAZA LA HIPOTESIS NULA H0")
    print(
        "No existe evidencia suficiente para "
        "afirmar que B produce mayor rendimiento."
    )

print("\n" + "=" * 65)
print("FASE 3 COMPLETADA")
print("=" * 65)



# ==================================================
# FASE 4: VISUALIZACIONES DEL PROYECTO INTEGRADOR
# ==================================================

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("\n" + "=" * 65)
print("FASE 4: GENERACION DE GRAFICAS")
print("=" * 65)

# Crear carpeta para guardar las graficas
carpeta_graficas = (
    Path(__file__).resolve().parent / "graficas"
)

carpeta_graficas.mkdir(
    parents=True,
    exist_ok=True
)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# GRAFICA 1: MATRIZ DE CORRELACION
# --------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    matriz_correlacion,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title("Matriz de correlacion de Pearson")
plt.tight_layout()

plt.savefig(
    carpeta_graficas / "01_correlacion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# GRAFICA 2: RIEGO VS RENDIMIENTO
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.regplot(
    data=df_parcelas,
    x="volumen_riego",
    y="rendimiento_ton",
    ci=95,
    scatter_kws={
        "alpha": 0.65,
        "s": 55
    },
    line_kws={
        "color": "red",
        "linewidth": 2
    }
)

plt.title("Volumen de riego vs. rendimiento")
plt.xlabel("Volumen de riego (m3/ha)")
plt.ylabel("Rendimiento (ton/ha)")
plt.tight_layout()

plt.savefig(
    carpeta_graficas / "02_regresion_riego.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# GRAFICA 3: COMPARACION DE TRATAMIENTOS
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df_parcelas,
    x="tipo_tratamiento",
    y="rendimiento_ton",
    order=[
        "Tratamiento_A",
        "Tratamiento_B"
    ],
    showmeans=True
)

plt.title("Rendimiento por tipo de tratamiento")
plt.xlabel("Tratamiento agricola")
plt.ylabel("Rendimiento (ton/ha)")
plt.tight_layout()

plt.savefig(
    carpeta_graficas / "03_boxplot_tratamientos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# GRAFICA 4: HISTOGRAMA DE RESIDUOS
# --------------------------------------------------

residuos_ols = modelo_ols.resid

plt.figure(figsize=(9, 6))

sns.histplot(
    residuos_ols,
    bins=15,
    kde=True
)

plt.axvline(
    x=0,
    color="red",
    linestyle="--",
    label="Residuo cero"
)

plt.title("Distribucion de residuos del modelo OLS")
plt.xlabel("Residuo (ton/ha)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()

plt.savefig(
    carpeta_graficas / "04_histograma_residuos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# VERIFICAR ARCHIVOS
# --------------------------------------------------

print("\nGRAFICAS GENERADAS:")

for archivo in sorted(
    carpeta_graficas.glob("*.png")
):
    print("-", archivo.name)

print("\nUBICACION:")
print(carpeta_graficas)

print("\n" + "=" * 65)
print("FASE 4 COMPLETADA")
print("=" * 65)
