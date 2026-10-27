"""Utilidades para analizar un diseño completamente al azar de un factor."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import f as distribucion_f
from scipy.stats import studentized_range
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.regression.linear_model import RegressionResultsWrapper


def cargar_datos(archivo: str | Path) -> pd.DataFrame:
    """Carga datos desde un CSV o un archivo de Excel.

    Args:
        archivo: Ruta del archivo CSV o Excel con extensión ``.xlsx``.

    Returns:
        Un DataFrame con los datos del archivo.

    Raises:
        ValueError: Si la extensión del archivo no es compatible.
    """
    # Se interpreta la extensión sin distinguir entre mayúsculas y minúsculas.
    ruta = Path(archivo)
    extension = ruta.suffix.lower()

    # Se deja que el detector de pandas identifique si el CSV usa coma o punto y coma.
    if extension == ".csv":
        return pd.read_csv(ruta, sep=None, engine="python")
    if extension == ".xlsx":
        return pd.read_excel(ruta)

    raise ValueError("El archivo debe tener formato CSV o Excel (.xlsx).")


def validar_dca(
    df: pd.DataFrame, col_trat: str, col_resp: str
) -> list[str]:
    """Valida las columnas y las réplicas para un DCA de un factor.

    Args:
        df: Datos que se analizarán.
        col_trat: Nombre de la columna con los tratamientos.
        col_resp: Nombre de la columna con la respuesta.

    Returns:
        Lista de errores en español; queda vacía si el DCA es válido.
    """
    errores: list[str] = []

    # Primero se comprueba que existan las variables necesarias para evitar accesos inválidos.
    columnas_faltantes = [
        columna for columna in (col_trat, col_resp) if columna not in df.columns
    ]
    for columna in columnas_faltantes:
        errores.append(f"No se encontró la columna '{columna}'.")
    if columnas_faltantes:
        return errores

    # El tratamiento y la respuesta deben representar variables distintas.
    if col_trat == col_resp:
        errores.append("Las columnas de tratamiento y respuesta deben ser distintas.")

    # Se verifica que la respuesta sea cuantitativa, condición para ajustar ANOVA.
    if not is_numeric_dtype(df[col_resp]) or is_bool_dtype(df[col_resp]):
        errores.append(f"La columna de respuesta '{col_resp}' debe ser numérica.")

    # Se detectan faltantes en cualquiera de las variables del análisis.
    if df[[col_trat, col_resp]].isna().any().any():
        errores.append("Hay datos faltantes en las columnas de tratamiento o respuesta.")

    # Se cuentan tratamientos y observaciones para comprobar replicación mínima.
    numero_tratamientos = df[col_trat].nunique(dropna=True)
    if numero_tratamientos < 2:
        errores.append("El DCA debe tener al menos 2 tratamientos.")

    # Para cada tratamiento se cuentan las respuestas observadas; cada grupo requiere n >= 2.
    replicas = df.groupby(col_trat, dropna=False)[col_resp].count()
    if (replicas < 2).any():
        errores.append("Cada tratamiento debe tener al menos 2 réplicas.")

    return errores


def ajustar_dca(
    df: pd.DataFrame, col_trat: str, col_resp: str
) -> RegressionResultsWrapper:
    """Ajusta el modelo lineal de un DCA de un factor.

    Args:
        df: Datos experimentales.
        col_trat: Nombre de la columna con los tratamientos.
        col_resp: Nombre de la columna con la respuesta.

    Returns:
        Modelo OLS ajustado.
    """
    # Q() protege nombres con espacios, tildes o caracteres especiales en la fórmula.
    respuesta = json.dumps(col_resp, ensure_ascii=False)
    tratamiento = json.dumps(col_trat, ensure_ascii=False)
    formula = f"Q({respuesta}) ~ C(Q({tratamiento}))"

    # Se ajusta Y_ij = μ + τ_i + ε_ij para estimar las medias por tratamiento.
    return ols(formula, data=df).fit()


def tabla_anova(modelo: RegressionResultsWrapper) -> pd.DataFrame:
    """Construye la tabla ANOVA tipo II y agrega la suma de cuadrados total.

    Args:
        modelo: Modelo OLS ajustado para un DCA de un factor.

    Returns:
        Tabla con fuente, grados de libertad, sumas y cuadrados medios, F y p-valor.
    """
    # La tabla tipo II particiona SC_Total = SC_Tratamientos + SC_Error.
    tabla = anova_lm(modelo, typ=2)
    fila_tratamientos = tabla.loc[tabla.index != "Residual"].iloc[0]
    fila_error = tabla.loc["Residual"]

    # Se calcula cada cuadrado medio como CM = SC / gl.
    gl_tratamientos = float(fila_tratamientos["df"])
    gl_error = float(fila_error["df"])
    sc_tratamientos = float(fila_tratamientos["sum_sq"])
    sc_error = float(fila_error["sum_sq"])
    filas = [
        {
            "Fuente": "Tratamientos",
            "gl": gl_tratamientos,
            "SC": sc_tratamientos,
            "CM": sc_tratamientos / gl_tratamientos,
            "F": float(fila_tratamientos["F"]),
            "p-valor": float(fila_tratamientos["PR(>F)"]),
        },
        {
            "Fuente": "Error",
            "gl": gl_error,
            "SC": sc_error,
            "CM": sc_error / gl_error,
            "F": np.nan,
            "p-valor": np.nan,
        },
    ]

    # La fila total suma las particiones y sus grados de libertad, sin prueba F asociada.
    gl_total = gl_tratamientos + gl_error
    sc_total = sc_tratamientos + sc_error
    filas.append(
        {
            "Fuente": "Total",
            "gl": gl_total,
            "SC": sc_total,
            "CM": sc_total / gl_total,
            "F": np.nan,
            "p-valor": np.nan,
        }
    )
    return pd.DataFrame(filas, columns=["Fuente", "gl", "SC", "CM", "F", "p-valor"])


def f_critica(tabla: pd.DataFrame, alfa: float = 0.05) -> float:
    """Calcula el valor crítico de F para tratamientos en un DCA.

    Args:
        tabla: Tabla ANOVA con las filas ``Tratamientos`` y ``Error``.
        alfa: Nivel de significancia entre 0 y 1.

    Returns:
        Valor crítico de la distribución F.

    Raises:
        ValueError: Si alfa o los grados de libertad no son válidos.
    """
    if not 0 < alfa < 1:
        raise ValueError("El nivel de significancia alfa debe estar entre 0 y 1.")

    # Los grados de libertad de tratamientos y error definen la distribución F bajo H0.
    filas = tabla.set_index("Fuente")
    if not {"Tratamientos", "Error"}.issubset(filas.index):
        raise ValueError("La tabla ANOVA debe incluir Tratamientos y Error.")
    gl_tratamientos = float(filas.loc["Tratamientos", "gl"])
    gl_error = float(filas.loc["Error", "gl"])
    if gl_tratamientos <= 0 or gl_error <= 0:
        raise ValueError("Los grados de libertad deben ser mayores que cero.")

    # El cuantil 1-alfa delimita la región de rechazo de la prueba F.
    return float(distribucion_f.ppf(1 - alfa, gl_tratamientos, gl_error))


def resumen_tratamientos(
    df: pd.DataFrame, col_trat: str, col_resp: str
) -> pd.DataFrame:
    """Resume el número de réplicas y la respuesta por tratamiento.

    Args:
        df: Datos experimentales.
        col_trat: Nombre de la columna con los tratamientos.
        col_resp: Nombre de la columna con la respuesta.

    Returns:
        DataFrame con n, total, media y desviación estándar por tratamiento.
    """
    # Se agrupan las observaciones para obtener n, Σy, media y s por tratamiento.
    resumen = df.groupby(col_trat, dropna=False)[col_resp].agg(
        n="count",
        total="sum",
        media="mean",
        **{"desviación estándar": "std"},
    )
    # La desviación se calcula como s = sqrt(Σ(y - media)^2 / (n - 1)).
    return resumen.reset_index()


def coeficiente_variacion(
    modelo: RegressionResultsWrapper, df: pd.DataFrame, col_resp: str
) -> float:
    """Calcula el coeficiente de variación del DCA.

    Args:
        modelo: Modelo OLS ajustado para el DCA.
        df: Datos experimentales.
        col_resp: Nombre de la columna con la respuesta.

    Returns:
        Coeficiente de variación porcentual.

    Raises:
        ValueError: Si la media general es cero y no se puede calcular el CV.
    """
    # El cuadrado medio del error estima la varianza residual: CME = Σe² / glE.
    cme = float(modelo.mse_resid)
    # La media general es el promedio de todas las observaciones: media = Σy / N.
    media_general = float(df[col_resp].mean())
    if media_general == 0:
        raise ValueError("No se puede calcular el CV porque la media general es cero.")

    # CV = sqrt(CME) / media general * 100 expresa el error relativo en porcentaje.
    return float(np.sqrt(cme) / media_general * 100)


def tukey(
    df: pd.DataFrame,
    col_trat: str,
    col_resp: str,
    alfa: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """Compara medias con Tukey y calcula una diferencia mínima significativa.

    Args:
        df: Datos experimentales.
        col_trat: Nombre de la columna con los tratamientos.
        col_resp: Nombre de la columna con la respuesta.
        alfa: Nivel de significancia.

    Returns:
        Una tupla con las comparaciones de Tukey en español y la DMS.

    Raises:
        ValueError: Si el nivel de significancia no está entre 0 y 1.
    """
    if not 0 < alfa < 1:
        raise ValueError("El nivel de significancia alfa debe estar entre 0 y 1.")

    # Tukey compara cada par de medias usando el error residual del modelo DCA.
    modelo = ajustar_dca(df, col_trat, col_resp)
    resultado = pairwise_tukeyhsd(
        endog=df[col_resp],
        groups=df[col_trat],
        alpha=alfa,
    )
    tabla_resultado = pd.DataFrame(
        resultado._results_table.data[1:],
        columns=resultado._results_table.data[0],
    ).rename(
        columns={
            "group1": "Tratamiento 1",
            "group2": "Tratamiento 2",
            "meandiff": "Diferencia de medias",
            "p-adj": "p-valor ajustado",
            "lower": "Límite inferior",
            "upper": "Límite superior",
            "reject": "Rechaza H0",
        }
    )

    # En el DCA balanceado, DMS = q(1-alfa; k; glE) * sqrt(CME / r).
    replicas = df.groupby(col_trat)[col_resp].count()
    numero_tratamientos = len(replicas)
    grados_libertad_error = float(modelo.df_resid)
    cme = float(modelo.mse_resid)
    balanceado = replicas.nunique() == 1
    if balanceado:
        replicas_efectivas = float(replicas.iloc[0])
    else:
        # Para tamaños distintos se usa la media armónica de r como aproximación de Tukey-Kramer.
        replicas_efectivas = float(numero_tratamientos / (1 / replicas).sum())
        warnings.warn(
            "El diseño no está balanceado; la DMS calculada con la media "
            "armónica de réplicas es aproximada.",
            UserWarning,
            stacklevel=2,
        )

    # El cuantil del rango studentizado da q crítico para alfa, k tratamientos y glE.
    q_critico = float(
        studentized_range.ppf(
            1 - alfa, numero_tratamientos, grados_libertad_error
        )
    )
    # DMS = q crítico * sqrt(CME / r), con r igual a réplicas (o media armónica).
    dms = q_critico * np.sqrt(cme / replicas_efectivas)
    return tabla_resultado, float(dms)


def interpretar_dca(tabla: pd.DataFrame, alfa: float = 0.05) -> str:
    """Interpreta la prueba F de tratamientos del DCA en lenguaje sencillo.

    Args:
        tabla: Tabla ANOVA con las columnas ``Fuente`` y ``p-valor``.
        alfa: Nivel de significancia para decidir la prueba.

    Returns:
        Interpretación de la decisión sobre la hipótesis nula.

    Raises:
        ValueError: Si no hay una fila de tratamientos con p-valor calculable.
    """
    # La hipótesis nula es H0: todas las medias de tratamientos son iguales.
    fila = tabla.loc[tabla["Fuente"] == "Tratamientos", "p-valor"]
    if fila.empty or pd.isna(fila.iloc[0]):
        raise ValueError("La tabla no contiene un p-valor válido para Tratamientos.")

    # Se rechaza H0 cuando p-valor < alfa; en caso contrario no hay evidencia suficiente.
    p_valor = float(fila.iloc[0])
    if p_valor < alfa:
        return (
            f"Se rechaza H0 (p-valor = {p_valor:.4g} < alfa = {alfa:g}). "
            "Hay evidencia de que al menos un tratamiento tiene una media distinta; "
            "considere las comparaciones múltiples para decidir entre tratamientos."
        )
    return (
        f"No se rechaza H0 (p-valor = {p_valor:.4g} >= alfa = {alfa:g}). "
        "No hay evidencia suficiente para afirmar que las medias de los "
        "tratamientos sean diferentes."
    )
