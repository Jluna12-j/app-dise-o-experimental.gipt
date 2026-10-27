"""Análisis de un diseño completamente al azar de un factor."""

from pathlib import Path
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.anova_utils import (
    ajustar_dca,
    cargar_datos,
    coeficiente_variacion,
    f_critica,
    interpretar_dca,
    resumen_tratamientos,
    tabla_anova,
    tukey,
    validar_dca,
)


@st.cache_data
def cargar_archivo(nombre: str, contenido: bytes | None = None) -> pd.DataFrame:
    """Carga desde la ruta predeterminada o desde un archivo enviado por el usuario.

    Args:
        nombre: Ruta local del archivo o nombre del archivo enviado.
        contenido: Bytes del archivo enviado; si es ``None``, se usa la ruta local.

    Returns:
        Datos cargados en un DataFrame.
    """
    if contenido is None:
        return cargar_datos(nombre)

    extension = Path(nombre).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=extension) as archivo_temporal:
        archivo_temporal.write(contenido)
        archivo_temporal.flush()
        return cargar_datos(archivo_temporal.name)


st.title("Diseño completamente al azar (DCA)")
st.markdown(
    """
    El DCA compara las medias de una respuesta entre tratamientos asignados
    aleatoriamente a unidades experimentales homogéneas. En este ejemplo se
    evalúa la humedad final obtenida con **cuatro secadores solares (A, B, C y D)**,
    con cuatro réplicas por secador.
    """
)

archivo_cargado = st.file_uploader(
    "Cargue los datos en formato CSV o Excel (.xlsx)",
    type=["csv", "xlsx"],
)

try:
    if archivo_cargado is None:
        ruta_ejemplo = Path(__file__).resolve().parents[1] / "data" / "dca_secadores.csv"
        df = cargar_archivo(str(ruta_ejemplo))
        st.caption("No se cargó un archivo; se utilizan los datos de ejemplo de data/dca_secadores.csv.")
    else:
        df = cargar_archivo(archivo_cargado.name, archivo_cargado.getvalue())
except (OSError, ValueError, UnicodeDecodeError) as error:
    st.error(f"No fue posible cargar el archivo: {error}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("El archivo debe contener observaciones y al menos dos columnas.")
    st.stop()

columnas = list(df.columns)
columnas_numericas = list(df.select_dtypes(include="number").columns)
columnas_tratamiento = [columna for columna in columnas if columna not in columnas_numericas]
if not columnas_tratamiento:
    columnas_tratamiento = columnas
columnas_respuesta = [
    columna
    for columna in columnas_numericas
    if str(columna).strip().lower() not in {"replica", "réplica", "rep", "id"}
] or columnas_numericas

col_trat = st.selectbox(
    "Columna de tratamiento",
    columnas,
    index=columnas.index(columnas_tratamiento[0]),
)
col_resp = st.selectbox(
    "Columna de respuesta",
    columnas,
    index=columnas.index(columnas_respuesta[0]) if columnas_respuesta else 0,
)
alfa = st.slider(
    "Nivel de significancia (alfa)",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01,
    format="%.2f",
)

errores = validar_dca(df, col_trat, col_resp)
if errores:
    for error in errores:
        st.error(error)
    st.stop()

# El modelo y las cantidades derivadas se calculan una sola vez para mantener consistencia entre pestañas.
modelo = ajustar_dca(df, col_trat, col_resp)
tabla = tabla_anova(modelo)
resumen = resumen_tratamientos(df, col_trat, col_resp)
cv = coeficiente_variacion(modelo, df, col_resp)
valor_f = f_critica(tabla, alfa)
p_valor = float(tabla.loc[tabla["Fuente"] == "Tratamientos", "p-valor"].iloc[0])

pestanas = st.tabs(["Datos", "ANOVA", "Comparaciones", "Gráficos"])

with pestanas[0]:
    st.subheader("Datos experimentales")
    st.dataframe(df, width="stretch", hide_index=True)
    st.subheader("Resumen por tratamiento")
    st.dataframe(
        resumen.style.format(
            {
                "total": "{:.4f}",
                "media": "{:.4f}",
                "desviación estándar": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with pestanas[1]:
    st.subheader("Análisis de varianza")
    st.dataframe(
        tabla.style.format(
            {
                "gl": "{:.4f}",
                "SC": "{:.4f}",
                "CM": "{:.4f}",
                "F": "{:.4f}",
                "p-valor": "{:.4f}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )
    st.metric("F crítica", f"{valor_f:.4f}")
    st.metric("Coeficiente de variación (CV)", f"{cv:.4f}%")
    interpretacion = interpretar_dca(tabla, alfa)
    if p_valor < alfa:
        st.success(interpretacion)
    else:
        st.info(interpretacion)

with pestanas[2]:
    st.subheader("Comparaciones múltiples de Tukey")
    if p_valor < alfa:
        # La prueba de Tukey identifica qué pares difieren después de detectar diferencias en el ANOVA.
        tabla_tukey, dms = tukey(df, col_trat, col_resp, alfa)
        st.dataframe(
            tabla_tukey.style.format(
                {
                    "Diferencia de medias": "{:.4f}",
                    "p-valor ajustado": "{:.4f}",
                    "Límite inferior": "{:.4f}",
                    "Límite superior": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.metric("Diferencia mínima significativa (DMS)", f"{dms:.4f}")
    else:
        st.info(
            f"No se realiza Tukey porque el ANOVA no detectó diferencias "
            f"significativas entre tratamientos (p-valor = {p_valor:.4f} "
            f"≥ alfa = {alfa:.2f})."
        )

with pestanas[3]:
    st.subheader("Distribución de la respuesta")
    figura_cajas = px.box(
        df,
        x=col_trat,
        y=col_resp,
        points="all",
        labels={col_trat: "Tratamiento", col_resp: "Respuesta"},
        title="Respuesta por tratamiento",
    )
    st.plotly_chart(figura_cajas, width="stretch")

    st.subheader("Media de respuesta por tratamiento")
    figura_medias = px.bar(
        resumen,
        x=col_trat,
        y="media",
        labels={col_trat: "Tratamiento", "media": "Media de la respuesta"},
        title="Medias por tratamiento",
    )
    st.plotly_chart(figura_medias, width="stretch")
