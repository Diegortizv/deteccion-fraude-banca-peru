# Detección de clientes con fraude en la banca digital peruana mediante ensembles balanceados

Proyecto final del curso **IA Fundamentals** (Maestría en Ciencia de Datos e IA, UTEC). Este
repositorio contiene todo lo necesario para reproducir los resultados del informe: el código, los
resultados de los experimentos, las figuras y el informe en LaTeX.

**Pregunta:** con el mismo preprocesamiento, las mismas particiones y el mismo ensemble balanceado,
¿qué familia de modelo base ordena mejor a los clientes por riesgo de fraude?

**Idea central:** todo el proceso queda fijo (datos, preprocesamiento, 25 folds, ensemble de 30
submuestras 1:1 y métricas) y solo cambia el modelo base, para que cualquier diferencia se pueda
atribuir al modelo.

## Resultados

Media ± desviación estándar en los 25 folds de prueba (`resultados/metricas_*.csv`). El rango medio
se calcula en PR-AUC entre E1 y E4 (1 = mejor).

| Exp. | Modelo base | PR-AUC | ROC-AUC | Recall top 5 % | Precisión top 5 % | Rango medio |
|---|---|---|---|---|---|---|
| E0 | Dummy (azar, sin ensemble) | 0,035 ± 0,001 | 0,496 ± 0,012 | 0,037 ± 0,028 | 0,026 ± 0,019 | — |
| E1 | Regresión logística (caso base) | **0,168 ± 0,040** | 0,824 ± 0,031 | **0,266 ± 0,048** | **0,185 ± 0,033** | **1,60** |
| E2 | Árbol de decisión | 0,104 ± 0,021 | 0,791 ± 0,027 | 0,171 ± 0,052 | 0,119 ± 0,036 | 3,92 |
| E3 | KNN (k = 15) | 0,147 ± 0,025 | 0,827 ± 0,024 | 0,248 ± 0,049 | 0,173 ± 0,034 | 2,40 |
| E4 | XGBoost | 0,151 ± 0,031 | 0,824 ± 0,026 | 0,251 ± 0,066 | 0,175 ± 0,046 | 2,08 |

Friedman: χ² = 45,19, p = 8,4 × 10⁻¹⁰. Según Nemenyi (diferencia crítica = 0,94), el árbol es
significativamente peor que los otros tres, y regresión logística, XGBoost y KNN no se separan entre
sí. Según Wilcoxon con corrección de Holm, los tres modelos quedan por debajo de la regresión
logística, aunque frente a XGBoost y KNN la diferencia es pequeña (unos 0,02 de PR-AUC).

## Estructura

```
.
├── config.py               # Condiciones fijas: rutas, columnas, semilla, folds, K = 30, métricas
├── comun.py                # Carga y verificación de datos, pipeline, particiones y métricas
├── modelos.py              # Los 5 experimentos (E0–E4) con hiperparámetros fijos
├── verificar_entorno.py    # Comprueba versiones, huella del Excel y que los 5 modelos funcionen
├── experimentos.py         # Corre los experimentos y guarda resultados/metricas_*.csv y predicciones_*.csv
├── analisis.py             # Tabla principal, Friedman/Nemenyi/Wilcoxon-Holm y figuras base
├── figuras_informe.py      # Figuras del informe y de las diapositivas
├── cifras_informe.py       # Recalcula las demás cifras citadas en el informe
├── 01_experimentos.ipynb   # Corre E0–E4 desde Jupyter (guardado con sus salidas)
├── 02_analisis.ipynb       # Tabla, pruebas estadísticas y figuras base (guardado con sus salidas)
├── requirements.txt        # Versiones exactas de las librerías
├── datos/                  # Aquí va el Excel (no se incluye; ver "Datos")
├── resultados/             # CSV de los 5 experimentos (25 folds cada uno) y registro del entorno
├── figuras/                # Figuras generadas
└── informe/                # Informe en LaTeX (IEEE, 4 páginas) y su bibliografía
```

`config.py`, `comun.py` y `modelos.py` definen el protocolo común del equipo y no se modificaron
durante los experimentos.

## Datos

Conjunto **Data fraude bancario en Perú** (Kaggle, usuario eddisoncallatamamani):
https://www.kaggle.com/datasets/eddisoncallatamamani/data-fraude-bancario-en-per

- 5 960 clientes, 27 columnas, 209 con fraude confirmado (3,51 %). Cada fila es un cliente.
- **El archivo no se incluye en el repositorio** porque la página no declara una licencia.
  Descárgalo y guárdalo como `datos/DATOS_OFICIALES.xlsx`.
- El código verifica la huella SHA-256 del archivo y se detiene si no coincide:
  `8cdec41aa0a8bb4832000817d45e160897b30dcaa91cc5ad87b4d10a1867a0c7`

## Cómo reproducir

Requiere Python 3.10, 3.11 o 3.12. Los resultados del informe se generaron con Python 3.12.13 en
macOS (Apple M5); `resultados/requirements_lock.txt` registra el entorno exacto.

```bash
# 1. Entorno con las versiones exactas
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# En Mac, si xgboost no carga por libomp:  brew install libomp

# 2. Datos: descargar el Excel de Kaggle y guardarlo como datos/DATOS_OFICIALES.xlsx

# 3. Verificar entorno y datos (todo debe decir OK)
python verificar_entorno.py

# 4. Correr los 5 experimentos (~25 s en total en un Apple M5)
python experimentos.py todos

# 5. Tabla, pruebas estadísticas y figuras base (también se puede abrir y correr en Jupyter)
jupyter nbconvert --to notebook --execute --inplace 02_analisis.ipynb

# 6. Figuras y cifras del informe
python figuras_informe.py
python cifras_informe.py

# 7. Compilar el informe (con Tectonic: brew install tectonic)
cd informe && tectonic main.tex
```

Con las mismas versiones y la misma semilla, los CSV son idénticos: dos corridas en la misma
máquina dieron archivos iguales byte a byte, y las cifras coinciden con las obtenidas por otro
integrante en otra computadora. Entre sistemas operativos distintos puede variar el último decimal.

**Nota para macOS:** la regresión logística imprime muchos `RuntimeWarning ... encountered in
matmul`. Son avisos falsos de Accelerate, la librería de álgebra lineal de macOS: los resultados
son finitos e iguales a un cálculo sin BLAS. No afectan las métricas.

## De dónde sale cada elemento del informe

| Elemento del informe | Se genera con | Archivo |
|---|---|---|
| Tabla III (resultados) | `analisis.tabla_resumen` (`02_analisis.ipynb`) | `resultados/metricas_*.csv` |
| Tabla IV y pruebas estadísticas | `analisis.pruebas_estadisticas` (`02_analisis.ipynb`) | `resultados/metricas_*.csv` |
| Fig. 1 (pipeline) | Diagrama del equipo | `informe/figuras/pipeline.png` |
| Fig. 2 (boxplot de PR-AUC) | `figuras_informe.py` | `figuras/boxplot_comparado.pdf` |
| Fig. 3 (curvas de captura por modelo) | `figuras_informe.py` | `figuras/captura_paneles.pdf` |
| Cifras del dataset, de los folds, del ejemplo del fold 1, de las bandas y de los empates del árbol | `cifras_informe.py` | salida en pantalla |

`figuras_informe.py` también genera las figuras por modelo para las diapositivas
(`captura_<modelo>.png` y `pr_<modelo>.png`), y `analisis.py` genera `boxplot_pr_auc.png`,
`curvas_pr.png` y `diagrama_cd.png`.

## Protocolo fijo

| Componente | Valor |
|---|---|
| Columnas excluidas | `fecha_registro` (artificial), `alerta_sistema` y `score_riesgo` (salidas de otro sistema antifraude: *leakage*) |
| Predictores | 23: 19 numéricos y 4 categóricos |
| Preprocesamiento | `log1p` a ingreso y montos; estandarización; *one-hot*. Se ajusta solo con el fold de entrenamiento |
| Validación | `RepeatedStratifiedKFold(5, 5, random_state=42)`: 25 folds |
| Ensemble | `BalancedBaggingClassifier`: 30 submuestras 1:1, sin reemplazo, promedio de probabilidades |
| Métrica principal | PR-AUC (`average_precision`) |
| Métricas secundarias | ROC-AUC; recall y precisión en el 5 % de clientes con mayor riesgo |
| Comparación | Friedman + Nemenyi; Wilcoxon pareado contra el caso base con corrección de Holm |
| Semilla | 42 |

## Equipo

Junior Godofredo Baldeon Blas, Adnan Ahmed Choudary Ahmed, Eduardo Sebastián Lévano Cavero,
Michelle Yunorvi Luna Alama y Diego Alonso Ortiz Vera.

Docente: Vicente Machaca Arceda.
