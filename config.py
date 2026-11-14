"""
Condiciones iniciales del proyecto (caso base).

NO MODIFICAR. Todos los experimentos corren con estos valores para que la única
diferencia entre ellos sea el modelo base. Si algo tiene que cambiar, se cambia
aquí, una sola vez, y se avisa a todo el equipo antes de volver a correr.
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DATA_PATH = RAIZ / "datos" / "DATOS_OFICIALES.xlsx"
RESULTADOS_DIR = RAIZ / "resultados"
FIGURAS_DIR = RAIZ / "figuras"

# Huella SHA-256 del Excel. Si tu archivo no coincide, no estás usando los mismos datos.
DATA_SHA256 = "8cdec41aa0a8bb4832000817d45e160897b30dcaa91cc5ad87b4d10a1867a0c7"
N_FILAS_ESPERADAS = 5960
N_FRAUDES_ESPERADOS = 209

# --- Datos -------------------------------------------------------------------
TARGET = "fraude_confirmado"
# fecha_registro: fecha artificial sin señal. alerta_sistema y score_riesgo: salidas
# de otro sistema antifraude (leakage).
EXCLUIR = ["fecha_registro", "alerta_sistema", "score_riesgo"]
CATEGORICAS = ["departamento", "banco_principal", "app_preferida", "tipo_operacion_frecuente"]
LOG1P = ["ingreso_mensual", "monto_operacion", "monto_promedio_transaccion"]

# --- Validación --------------------------------------------------------------
SEED = 42
N_SPLITS = 5          # 5 folds estratificados...
N_REPEATS = 5         # ...repetidos 5 veces = 25 folds, idénticos para todos los modelos

# --- Resampling + ensemble ---------------------------------------------------
K_SUBMODELOS = 30         # número de submuestras balanceadas (un modelo por submuestra)
SAMPLING_STRATEGY = 1.0   # 1:1 fraude : no-fraude en cada submuestra

# --- Evaluación --------------------------------------------------------------
TOP_K = 0.05                                      # recall y precisión en el top 5% de riesgo
MODELOS_COMPARADOS = ["logreg", "arbol", "knn", "xgboost"]   # entran a las pruebas estadísticas
CASO_BASE = "logreg"
ALPHA = 0.05
