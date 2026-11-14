"""
Corre uno o varios experimentos y guarda los resultados en resultados/.

Uso desde la terminal:
    python experimentos.py logreg          # un experimento
    python experimentos.py arbol knn       # varios
    python experimentos.py todos           # E0 a E4

Uso desde un notebook:
    from experimentos import correr
    correr("logreg")
"""
import sys
import time

import config as C
from comun import cargar_datos, evaluar
from modelos import obtener_modelos


def correr(nombre, verbose=True):
    modelos = obtener_modelos()
    if nombre not in modelos:
        raise ValueError(f"Experimento desconocido: {nombre}. Opciones: {list(modelos)}")
    X, y = cargar_datos()
    modelo, usa_ensemble = modelos[nombre]

    print(f"Experimento '{nombre}' ({'ensemble x' + str(C.K_SUBMODELOS) if usa_ensemble else 'sin ensemble'})")
    t0 = time.time()
    metricas, predicciones = evaluar(nombre, modelo, usa_ensemble, X, y, verbose=verbose)

    C.RESULTADOS_DIR.mkdir(exist_ok=True)
    metricas.to_csv(C.RESULTADOS_DIR / f"metricas_{nombre}.csv", index=False)
    predicciones.to_csv(C.RESULTADOS_DIR / f"predicciones_{nombre}.csv", index=False)

    resumen = metricas[["pr_auc", "roc_auc", "recall_top5", "precision_top5"]].agg(["mean", "std"])
    print(f"Listo en {time.time() - t0:.0f} s. Media ± DE sobre {len(metricas)} folds:")
    print(resumen.round(3).to_string())
    return metricas


if __name__ == "__main__":
    argumentos = sys.argv[1:] or ["todos"]
    nombres = list(obtener_modelos()) if argumentos == ["todos"] else argumentos
    for n in nombres:
        correr(n)
