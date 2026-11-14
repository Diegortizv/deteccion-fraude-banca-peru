"""
Junta los resultados de todos los experimentos: tabla principal, pruebas
estadísticas y figuras. Se usa desde 02_analisis.ipynb.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import precision_recall_curve

import config as C

METRICAS = ["pr_auc", "roc_auc", "recall_top5", "precision_top5"]
ETIQUETAS = {"dummy": "E0 Dummy", "logreg": "E1 Reg. logística (base)", "arbol": "E2 Árbol",
             "knn": "E3 KNN", "xgboost": "E4 XGBoost"}
# Valores críticos de Nemenyi para alfa = 0.05 (Demšar, 2006, tabla 5a), según nº de modelos k
Q_NEMENYI_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def cargar_resultados():
    archivos = sorted(C.RESULTADOS_DIR.glob("metricas_*.csv"))
    if not archivos:
        raise FileNotFoundError("No hay resultados en resultados/. Corre primero los experimentos.")
    return pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)


def verificar_resultados(r):
    """Confirma que cada modelo tiene los 25 folds y que todos usaron las mismas particiones."""
    n_folds = C.N_SPLITS * C.N_REPEATS
    conteo = r.groupby("modelo").fold.nunique()
    incompletos = conteo[conteo != n_folds]
    faltantes = [m for m in ["dummy"] + C.MODELOS_COMPARADOS if m not in conteo.index]
    if len(incompletos):
        print("ATENCIÓN, modelos con folds incompletos:", incompletos.to_dict())
    if faltantes:
        print("ATENCIÓN, faltan experimentos:", faltantes)
    # mismas particiones => mismo tamaño y mismo nº de fraudes en cada fold, para todos los modelos
    firmas = r.pivot(index="fold", columns="modelo", values="fraudes_test")
    if not (firmas.nunique(axis=1) == 1).all():
        print("ATENCIÓN: los modelos no usaron las mismas particiones.")
    elif not len(incompletos) and not faltantes:
        print(f"OK: {conteo.size} modelos x {n_folds} folds, mismas particiones para todos.")


def tabla_resumen(r):
    """Media ± desviación estándar por modelo y rango medio en PR-AUC (1 = mejor)."""
    agg = r.groupby("modelo")[METRICAS].agg(["mean", "std"])
    tabla = pd.DataFrame({m: agg[m]["mean"].map("{:.3f}".format) + " ± " + agg[m]["std"].map("{:.3f}".format)
                          for m in METRICAS})
    piv = r.pivot(index="fold", columns="modelo", values="pr_auc")
    comparados = [m for m in C.MODELOS_COMPARADOS if m in piv.columns]
    rangos = piv[comparados].rank(axis=1, ascending=False).mean()
    tabla["rango_medio"] = rangos.reindex(tabla.index).map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    orden = [m for m in ["dummy"] + C.MODELOS_COMPARADOS if m in tabla.index]
    tabla = tabla.loc[orden]
    tabla.index = [ETIQUETAS.get(m, m) for m in tabla.index]
    tabla.columns = ["PR-AUC", "ROC-AUC", "Recall top 5%", "Precisión top 5%", "Rango medio"]
    return tabla


def holm(pvalores):
    """Corrección de Holm-Bonferroni. Devuelve p-valores ajustados en el orden original."""
    p = np.asarray(pvalores, dtype=float)
    m, orden = len(p), np.argsort(p)
    ajustados = np.empty(m)
    acumulado = 0.0
    for i, idx in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - i) * p[idx]))
        ajustados[idx] = acumulado
    return ajustados


def pruebas_estadisticas(r, metrica="pr_auc"):
    piv = r.pivot(index="fold", columns="modelo", values=metrica)[C.MODELOS_COMPARADOS]
    n, k = piv.shape
    fried = friedmanchisquare(*[piv[m] for m in C.MODELOS_COMPARADOS])
    rangos = piv.rank(axis=1, ascending=False).mean().sort_values()
    cd = Q_NEMENYI_005[k] * np.sqrt(k * (k + 1) / (6 * n))
    pares = pd.DataFrame([dict(modelo_a=a, modelo_b=b, dif_rango=abs(rangos[a] - rangos[b]),
                               significativo=abs(rangos[a] - rangos[b]) > cd)
                          for a, b in combinations(rangos.index, 2)])
    otros = [m for m in C.MODELOS_COMPARADOS if m != C.CASO_BASE]
    p_crudos = [wilcoxon(piv[m], piv[C.CASO_BASE]).pvalue for m in otros]
    vs_base = pd.DataFrame(dict(
        modelo=otros,
        dif_media=[piv[m].mean() - piv[C.CASO_BASE].mean() for m in otros],
        p_wilcoxon=p_crudos,
        p_holm=holm(p_crudos),
    ))
    vs_base["significativo"] = vs_base.p_holm < C.ALPHA
    return dict(metrica=metrica, n_folds=n, friedman=fried, rangos=rangos, cd=cd,
                pares_nemenyi=pares, vs_caso_base=vs_base)


def imprimir_pruebas(res):
    f = res["friedman"]
    print(f"Métrica: {res['metrica']}  ·  {res['n_folds']} folds")
    print(f"Friedman: chi2 = {f.statistic:.3f}, p = {f.pvalue:.4g}  "
          f"-> {'hay diferencias' if f.pvalue < C.ALPHA else 'no hay diferencias significativas'}")
    print("\nRango medio (1 = mejor):")
    print(res["rangos"].round(2).to_string())
    print(f"\nDiferencia crítica de Nemenyi (alfa = {C.ALPHA}): {res['cd']:.3f}")
    print(res["pares_nemenyi"].round(3).to_string(index=False))
    print(f"\nWilcoxon pareado contra el caso base ({C.CASO_BASE}), corrección de Holm:")
    print(res["vs_caso_base"].round(4).to_string(index=False))
    if f.pvalue >= C.ALPHA:
        print("\nComo Friedman no es significativo, las comparaciones por pares son solo descriptivas.")


def _guardar(fig, nombre):
    C.FIGURAS_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIGURAS_DIR / nombre, dpi=300, bbox_inches="tight")


def figura_boxplot(r, metrica="pr_auc"):
    orden = [m for m in ["dummy"] + C.MODELOS_COMPARADOS if m in r.modelo.unique()]
    datos = [r.loc[r.modelo == m, metrica] for m in orden]
    prevalencia = C.N_FRAUDES_ESPERADOS / C.N_FILAS_ESPERADAS
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(datos, showmeans=True)
    ax.set_xticks(range(1, len(orden) + 1), [ETIQUETAS[m] for m in orden], rotation=15)
    ax.axhline(prevalencia, ls="--", lw=1, color="gray", label=f"Línea base (prevalencia = {prevalencia:.3f})")
    ax.set_ylabel("PR-AUC por fold" if metrica == "pr_auc" else metrica)
    ax.set_title(f"Distribución de {metrica} en los {r.fold.nunique()} folds")
    ax.legend(loc="upper left", fontsize=8)
    _guardar(fig, f"boxplot_{metrica}.png")
    return fig


def figura_curvas_pr(repeticion=0):
    """Curvas PR con las predicciones de los test folds de una repetición (cada cliente aparece una vez)."""
    folds = range(repeticion * C.N_SPLITS, (repeticion + 1) * C.N_SPLITS)
    prevalencia = C.N_FRAUDES_ESPERADOS / C.N_FILAS_ESPERADAS
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in C.MODELOS_COMPARADOS:
        ruta = C.RESULTADOS_DIR / f"predicciones_{m}.csv"
        if not ruta.exists():
            continue
        p = pd.read_csv(ruta)
        p = p[p.fold.isin(folds)]
        precision, recall, _ = precision_recall_curve(p.y_real, p.prob)
        ax.plot(recall, precision, lw=1.4, label=ETIQUETAS[m])
    ax.axhline(prevalencia, ls="--", lw=1, color="gray", label="Azar (prevalencia)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"Curvas precisión-recall (repetición {repeticion + 1})")
    ax.legend(fontsize=8)
    _guardar(fig, "curvas_pr.png")
    return fig


def figura_cd(res):
    """Diagrama de diferencia crítica: los modelos unidos por una barra no difieren significativamente."""
    rangos, cd, k = res["rangos"], res["cd"], len(res["rangos"])
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.set_xlim(0.8, k + 0.2); ax.set_ylim(-1.6, 1.2); ax.axis("off")
    ax.plot([1, k], [0, 0], color="black", lw=1)
    for v in range(1, k + 1):
        ax.plot([v, v], [0, 0.08], color="black", lw=1)
        ax.text(v, 0.15, str(v), ha="center", fontsize=8)
    ax.plot([1, 1 + cd], [0.75, 0.75], color="black", lw=2)
    ax.text(1 + cd / 2, 0.85, f"CD = {cd:.2f}", ha="center", fontsize=8)
    for i, (m, rk) in enumerate(rangos.items()):
        y_txt = -0.45 - 0.35 * i
        ax.plot([rk, rk], [0, y_txt], color="gray", lw=0.8)
        ax.plot(rk, 0, "o", color="black", ms=4)
        ax.text(rk + 0.03, y_txt, f"{ETIQUETAS[m]} ({rk:.2f})", va="center", fontsize=8)
    # grupos de modelos consecutivos cuya diferencia de rango no supera CD
    valores, grupos, y_bar = rangos.values, [], -0.18
    for i in range(k):
        j = max(jj for jj in range(i, k) if valores[jj] - valores[i] <= cd)
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for n_g, (i, j) in enumerate(grupos):
        y = y_bar - 0.1 * n_g
        ax.plot([valores[i] - 0.03, valores[j] + 0.03], [y, y], color="#c8891a", lw=3)
    ax.set_title(f"Diagrama de diferencia crítica (Nemenyi, alfa = {C.ALPHA}) · {res['metrica']}", fontsize=9)
    _guardar(fig, "diagrama_cd.png")
    return fig
