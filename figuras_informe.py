"""
Figuras complementarias para el informe y las diapositivas (estilo de las figuras de Michelle).

Uso:   python figuras_informe.py

Lee resultados/predicciones_<modelo>.csv y resultados/metricas_<modelo>.csv y guarda en figuras/:
  - captura_<modelo>.png     % de fraudes encontrados según el % de clientes revisados (25 folds)
  - pr_<modelo>.png          curvas precisión-recall de las 5 repeticiones
  - captura_paneles.png/.pdf     una curva de captura por modelo con su banda entre folds (informe, Fig. 3)
  - captura_comparada.png/.pdf   las 4 curvas de captura promedio juntas
  - pr_comparada.png/.pdf        las 4 curvas PR de la repetición 1 juntas (para el informe)
  - boxplot_comparado.png/.pdf   PR-AUC de los 25 folds por modelo, con la línea de azar (para el informe)

No cambia ninguna condición experimental: solo lee los CSV que genera experimentos.py.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import precision_recall_curve

import config as C

MODELOS = C.MODELOS_COMPARADOS  # logreg, arbol, knn, xgboost (orden fijo)
NOMBRES = {"logreg": "E1 Regresión logística", "arbol": "E2 Árbol de decisión",
           "knn": "E3 KNN", "xgboost": "E4 XGBoost"}
# Paleta categórica en orden fijo (un color por modelo, nunca por rango) + estilo de línea
# como segunda codificación, para que se distingan también en blanco y negro.
COLOR = {"logreg": "#2a78d6", "arbol": "#eb6834", "knn": "#1baf7a", "xgboost": "#eda100"}
LINEA = {"logreg": "-", "arbol": "--", "knn": "-.", "xgboost": ":"}
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#9a9893"
PREVALENCIA = C.N_FRAUDES_ESPERADOS / C.N_FILAS_ESPERADAS
MARCAS = (0.05, 0.20)  # fracciones de clientes revisados que se anotan
REJILLA = np.linspace(0, 1, 401)

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2,
    "ytick.color": TINTA_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False,
})
coma = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
porc = FuncFormatter(lambda v, _: f"{v:.0f} %")


def _pct(v, dec=0):
    return f"{100 * v:.{dec}f}".replace(".", ",") + " %"


def _guardar(fig, nombre, pdf=False):
    C.FIGURAS_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIGURAS_DIR / f"{nombre}.png", dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(C.FIGURAS_DIR / f"{nombre}.pdf", bbox_inches="tight")
    plt.close(fig)


def cargar_predicciones(m):
    return pd.read_csv(C.RESULTADOS_DIR / f"predicciones_{m}.csv")


def curvas_captura(pred):
    """Por fold: fracción de fraudes encontrados al revisar la fracción x de clientes, de mayor a
    menor riesgo (mismo orden estable que comun.metricas_top_k). Devuelve una fila por fold."""
    filas, marcas = [], {q: [] for q in MARCAS}
    for _, d in pred.groupby("fold"):
        orden = np.argsort(-d.prob.to_numpy(), kind="stable")
        y = d.y_real.to_numpy()[orden]
        n, total = len(y), y.sum()
        x = np.arange(n + 1) / n
        capt = np.concatenate([[0], np.cumsum(y)]) / total
        filas.append(np.interp(REJILLA, x, capt))
        for q in MARCAS:  # valor exacto, igual que recall_top5 en metricas_<modelo>.csv
            marcas[q].append(capt[int(np.ceil(q * n))])
    return np.array(filas), {q: float(np.mean(v)) for q, v in marcas.items()}


def figura_captura_modelo(m):
    curvas, marcas = curvas_captura(cargar_predicciones(m))
    x = 100 * REJILLA
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.fill_between(x, 100 * np.percentile(curvas, 10, axis=0), 100 * np.percentile(curvas, 90, axis=0),
                    color=COLOR[m], alpha=0.15, lw=0, label="Rango entre folds (percentiles 10–90)")
    ax.plot(x, 100 * curvas.mean(axis=0), color=COLOR[m], lw=2, label="Promedio de los 25 folds")
    ax.plot([0, 100], [0, 100], color=GRIS, lw=1, ls="--", label="Azar")
    for q, v in marcas.items():
        ax.plot(100 * q, 100 * v, "o", ms=6, color=COLOR[m], mec="white", mew=1.5, zorder=3)
        ax.annotate(f"revisando {_pct(q)} → {_pct(v)} de los fraudes", (100 * q, 100 * v),
                    xytext=(8, -4), textcoords="offset points", fontsize=8, color=TINTA)
    ax.set(xlim=(0, 100), ylim=(0, 101), xlabel="% de clientes revisados (de mayor a menor riesgo)",
           ylabel="% de fraudes encontrados", title=f"{NOMBRES[m]}: captura de fraudes")
    ax.xaxis.set_major_formatter(porc); ax.yaxis.set_major_formatter(porc)
    ax.legend(loc="upper left", fontsize=7)
    _guardar(fig, f"captura_{m}")


def figura_pr_modelo(m):
    pred = cargar_predicciones(m)
    met = pd.read_csv(C.RESULTADOS_DIR / f"metricas_{m}.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for rep in range(C.N_REPEATS):
        d = pred[pred.fold // C.N_SPLITS == rep]
        precision, recall, _ = precision_recall_curve(d.y_real, d.prob)
        primera = rep == 0
        ax.plot(recall, precision, color=COLOR[m], lw=1.6 if primera else 0.9, alpha=1 if primera else 0.45,
                label="Repetición 1" if primera else ("Repeticiones 2–5" if rep == 1 else None))
    r5, p5 = met.recall_top5.mean(), met.precision_top5.mean()
    ax.plot(r5, p5, "o", ms=6, color=COLOR[m], mec="white", mew=1.5, zorder=3)
    ax.annotate(f"Top 5 % (60 clientes)\nrecall {r5:.2f} · precisión {p5:.2f}".replace(".", ","), (r5, p5),
                xytext=(10, 18), textcoords="offset points", fontsize=8, color=TINTA,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.8))
    ax.axhline(PREVALENCIA, color=GRIS, lw=1, ls="--")
    ax.text(0.99, PREVALENCIA + 0.01, "azar (prevalencia)", ha="right", va="bottom", fontsize=7, color=TINTA_2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall (fraudes encontrados / fraudes totales)",
           ylabel="Precisión (fraudes / clientes revisados)", title=f"{NOMBRES[m]}: curvas precisión-recall")
    ax.xaxis.set_major_formatter(coma); ax.yaxis.set_major_formatter(coma)
    ax.legend(loc="upper right", fontsize=7)
    _guardar(fig, f"pr_{m}")


def figura_captura_comparada(hasta=0.5):
    """Las 4 curvas promedio en una figura de una columna (IEEE). Se corta en `hasta` porque más allá
    todas superan el 90 % y no se distinguen."""
    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    x = 100 * REJILLA
    for m in MODELOS:
        curvas, marcas = curvas_captura(cargar_predicciones(m))
        ax.plot(x, 100 * curvas.mean(axis=0), color=COLOR[m], ls=LINEA[m], lw=1.6,
                label=f"{NOMBRES[m]} ({_pct(marcas[0.05])} | {_pct(marcas[0.20])})")
    ax.plot([0, 100], [0, 100], color=GRIS, lw=0.8, ls="--", label="Azar")
    for q in MARCAS:
        ax.axvline(100 * q, color=GRIS, lw=0.6, ls=":")
    ax.set(xlim=(0, 100 * hasta), ylim=(0, 100), xlabel="% de clientes revisados (de mayor a menor riesgo)",
           ylabel="% de fraudes encontrados")
    ax.xaxis.set_major_formatter(porc); ax.yaxis.set_major_formatter(porc)
    ax.legend(loc="lower right", fontsize=5.5, title="Captura al 5 % | 20 %", title_fontsize=5.5,
              frameon=True, framealpha=0.95, edgecolor="none", handlelength=2.2, borderpad=0.3)
    _guardar(fig, "captura_comparada", pdf=True)


def figura_boxplot_comparado():
    """PR-AUC de los 25 folds por modelo, con el color de cada modelo y la línea de azar."""
    orden = ["dummy"] + MODELOS
    datos = [pd.read_csv(C.RESULTADOS_DIR / f"metricas_{m}.csv").pr_auc for m in orden]
    colores = [GRIS] + [COLOR[m] for m in MODELOS]
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    bp = ax.boxplot(datos, widths=0.5, patch_artist=True, showmeans=True,
                    medianprops=dict(color=TINTA, lw=1.2),
                    meanprops=dict(marker="D", ms=3.5, mfc="white", mec=TINTA, mew=0.8),
                    flierprops=dict(marker="o", ms=3, mfc="none", mec=TINTA_2),
                    whiskerprops=dict(color=TINTA_2, lw=0.8), capprops=dict(color=TINTA_2, lw=0.8))
    for caja, c in zip(bp["boxes"], colores):
        caja.set(facecolor=c, alpha=0.35, edgecolor=c, lw=1.2)
    ax.axhline(PREVALENCIA, color=GRIS, lw=0.8, ls="--")
    ax.text(5.45, PREVALENCIA + 0.004, "azar (0,035)", ha="right", va="bottom", fontsize=6, color=TINTA_2)
    ax.set_xticks(range(1, 6), ["E0\nDummy", "E1\nReg. log.", "E2\nÁrbol", "E3\nKNN", "E4\nXGBoost"])
    ax.set_ylabel("PR-AUC por fold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.grid(axis="x", visible=False)
    _guardar(fig, "boxplot_comparado", pdf=True)


def figura_pr_comparada(repeticion=0):
    """Curvas PR de los 4 modelos con las predicciones de una repetición (cada cliente aparece una vez)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in MODELOS:
        pred = cargar_predicciones(m)
        d = pred[pred.fold // C.N_SPLITS == repeticion]
        precision, recall, _ = precision_recall_curve(d.y_real, d.prob)
        ax.plot(recall, precision, color=COLOR[m], ls=LINEA[m], lw=1.3, label=NOMBRES[m])
    ax.axhline(PREVALENCIA, color=GRIS, lw=0.8, ls="--", label="Azar (prevalencia)")
    ax.set(xlim=(0, 1), ylim=(0, 0.6), xlabel="Recall", ylabel="Precisión")
    ax.xaxis.set_major_formatter(coma); ax.yaxis.set_major_formatter(coma)
    ax.legend(loc="upper right", fontsize=6)
    _guardar(fig, "pr_comparada", pdf=True)


def figura_captura_paneles():
    """Una curva de captura por modelo (2x2, ancho de una columna IEEE): promedio de los 25 folds y
    banda entre los percentiles 10 y 90 de los folds. Es el diseño de captura_<modelo>.png en formato
    compacto; los puntos marcan la captura exacta al revisar el 5 % y el 20 %."""
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.45), sharex=True, sharey=True)
    x = 100 * REJILLA
    for ax, m in zip(axs.flat, MODELOS):
        curvas, marcas = curvas_captura(cargar_predicciones(m))
        ax.fill_between(x, 100 * np.percentile(curvas, 10, axis=0), 100 * np.percentile(curvas, 90, axis=0),
                        color=COLOR[m], alpha=0.2, lw=0)
        ax.plot(x, 100 * curvas.mean(axis=0), color=COLOR[m], lw=1.4)
        ax.plot([0, 100], [0, 100], color=GRIS, lw=0.7, ls="--")
        for q, v in marcas.items():
            ax.plot(100 * q, 100 * v, "o", ms=3.5, color=COLOR[m], mec="white", mew=0.8, zorder=3)
            ax.annotate(_pct(v), (100 * q, 100 * v), xytext=(4, -7), textcoords="offset points",
                        fontsize=6, color=TINTA)
        ax.set_title(NOMBRES[m], fontsize=7, color=TINTA, pad=3)
        ax.set(xlim=(0, 100), ylim=(0, 101))
        ax.set_xticks([0, 50, 100]); ax.set_xticks([25, 75], minor=True)
        ax.set_yticks([0, 50, 100]); ax.set_yticks([25, 75], minor=True)
        ax.grid(which="minor", color="#e6e5e1", lw=0.6)
        ax.xaxis.set_major_formatter(porc); ax.yaxis.set_major_formatter(porc)
        ax.tick_params(labelsize=6); ax.tick_params(which="minor", length=0)
    fig.supxlabel("% de clientes revisados (de mayor a menor riesgo)", fontsize=7, color=TINTA_2, y=0.02)
    fig.supylabel("% de fraudes encontrados", fontsize=7, color=TINTA_2, x=0.02)
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
    _guardar(fig, "captura_paneles", pdf=True)


if __name__ == "__main__":
    for m in MODELOS:
        figura_captura_modelo(m)
        figura_pr_modelo(m)
    figura_captura_comparada()
    figura_captura_paneles()
    figura_pr_comparada()
    figura_boxplot_comparado()
    print(f"Figuras guardadas en {C.FIGURAS_DIR}")
