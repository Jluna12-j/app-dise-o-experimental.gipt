# Instrucciones de Copilot para este repositorio

## Contexto del proyecto

- Este proyecto es una aplicación de Diseño Experimental desarrollada en Python y Streamlit.
- Está dirigida al curso de Diseño Experimental de Ingeniería Agrícola de la Universidad de Sucre.

## Estructura del proyecto

- `app.py`: portada de la aplicación.
- `pages/`: un archivo por cada diseño experimental.
- `utils/`: funciones reutilizables; debe incluir `__init__.py`.
- `data/`: archivos CSV de ejemplo.
- `verificacion/`: scripts de R para verificación.

## Idioma y documentación

- Escribe todo el código, comentarios, docstrings y mensajes de la interfaz en español.
- Añade un comentario en cada paso estadístico que explique qué se calcula y por qué, con explicaciones adecuadas para estudiantes de Ingeniería Agrícola.
- Declara funciones con anotaciones de tipos (`type hints`) y docstrings en estilo Google, escritos en español.

## Dependencias

- Las únicas librerías permitidas son `streamlit`, `pandas`, `numpy`, `scipy`, `statsmodels`, `plotly` y `openpyxl`.
- No agregues otras dependencias sin pedir permiso.

## Análisis estadístico

- Ajusta los modelos de ANOVA con `statsmodels.formula.api.ols` y calcula las tablas con `statsmodels.stats.anova.anova_lm(typ=2)`.
- Realiza las comparaciones de Tukey con `statsmodels.stats.multicomp.pairwise_tukeyhsd`.
- No inventes ni codifiques resultados estadísticos de ejemplo como si fueran resultados reales: todos los números presentados deben provenir del cálculo realizado con los datos.

## Validación

- Antes de terminar una tarea, ejecuta el código pertinente y corrige los errores que aparezcan.