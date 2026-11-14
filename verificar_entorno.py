"""
Corre esto ANTES de cualquier experimento:   python verificar_entorno.py

Verifica que tengas las mismas versiones de librerías, el mismo Excel y que el
pipeline funcione con los cinco modelos. Si todo dice OK, tus resultados serán
iguales a los del resto del equipo.
"""
import importlib
import sys
import time

VERSIONES = {  # nombre de import -> versión esperada (ver requirements.txt)
    "numpy": "2.2.2",
    "pandas": "2.2.3",
    "scipy": "1.15.1",
    "sklearn": "1.6.1",
    "imblearn": "0.13.0",
    "xgboost": "2.1.4",
    "openpyxl": "3.1.5",
    "matplotlib": "3.10.0",
}

todo_ok = True


def reportar(ok, mensaje):
    global todo_ok
    todo_ok &= ok
    print(("  OK     " if ok else "  REVISAR ") + mensaje)


print("1) Python")
v = sys.version_info
reportar((3, 10) <= (v.major, v.minor) <= (3, 12), f"Python {v.major}.{v.minor}.{v.micro} (se recomienda 3.10 a 3.12)")

print("2) Librerías")
for modulo, esperada in VERSIONES.items():
    try:
        instalada = getattr(importlib.import_module(modulo), "__version__", "desconocida")
        reportar(instalada == esperada, f"{modulo} {instalada} (esperada {esperada})")
    except ImportError as e:
        reportar(False, f"{modulo} no se pudo importar: {e}")

if not todo_ok:
    print("\nInstala las versiones exactas con:  pip install -r requirements.txt")
    sys.exit(1)

import config as C
from comun import cargar_datos, construir_pipeline, particiones
from modelos import obtener_modelos

print("3) Datos")
try:
    X, y = cargar_datos()
    reportar(True, f"Excel correcto (SHA-256 coincide): {len(y)} clientes, {y.sum()} fraudes, {X.shape[1]} predictores")
except Exception as e:
    reportar(False, str(e))
    sys.exit(1)

print("4) Particiones")
tasas = [y[te].mean() for _, te in particiones().split(X, y)]
reportar(len(tasas) == C.N_SPLITS * C.N_REPEATS and all(0.03 <= t <= 0.04 for t in tasas),
         f"{len(tasas)} folds; tasa de fraude por test fold entre {min(tasas):.3%} y {max(tasas):.3%}")

print("5) Prueba rápida de cada modelo (un fold)")
tr, te = next(particiones().split(X, y))
for nombre, (modelo, usa_ensemble) in obtener_modelos().items():
    t0 = time.time()
    try:
        pipe = construir_pipeline(modelo, X, usa_ensemble).fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        reportar(p.shape == (len(te),) and 0 <= p.min() <= p.max() <= 1,
                 f"{nombre:8s} entrena y predice ({time.time() - t0:.1f} s)")
    except Exception as e:
        reportar(False, f"{nombre:8s} falló: {type(e).__name__}: {e}")

print("\n" + ("TODO OK: puedes correr los experimentos." if todo_ok
              else "Hay puntos por revisar antes de correr los experimentos."))
sys.exit(0 if todo_ok else 1)
