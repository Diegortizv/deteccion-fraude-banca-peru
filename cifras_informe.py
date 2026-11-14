"""
Recalcula las cifras del informe que no salen directamente de analisis.py.

Uso:   python cifras_informe.py

Necesita el Excel en datos/ y los CSV de resultados/ (python experimentos.py todos).
Imprime cada cifra con la sección del informe donde aparece. No cambia ninguna
condición experimental: usa las mismas funciones de comun.py.
"""
import os
import warnings

# Avisos espurios de BLAS (Accelerate) en macOS; la variable también llega a los procesos paralelos.
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score

import config as C
from comun import cargar_datos, construir_pipeline, construir_preprocesamiento, particiones
from figuras_informe import REJILLA, cargar_predicciones, curvas_captura
from modelos import obtener_modelos

warnings.filterwarnings("ignore", category=RuntimeWarning)
MODELOS = C.MODELOS_COMPARADOS
SEGURIDAD = ["tiene_token_fisico", "tiene_token_digital", "cambios_clave_recientes",
             "historial_suspicious", "ubicacion_inusual", "horario_inusual", "dispositivo_nuevo",
             "fallas_autenticacion", "intentos_contacto_sospechoso", "operaciones_canceladas"]


def titulo(texto):
    print(f"\n=== {texto} ===")


def tasa(df, col):
    return (df.groupby(col)[C.TARGET].mean() * 100).round(1).to_dict()


def main():
    df = pd.read_excel(C.DATA_PATH)
    X, y = cargar_datos()  # verifica la huella SHA-256
    metricas = {m: pd.read_csv(C.RESULTADOS_DIR / f"metricas_{m}.csv") for m in ["dummy"] + MODELOS}
    pred = {m: cargar_predicciones(m) for m in MODELOS}

    titulo("Datos (Introducción y Propuesta)")
    print(f"Clientes: {len(df)}, columnas: {df.shape[1]}, fraudes: {y.sum()} ({y.mean():.2%})")
    print(f"Valores nulos: {int(df.isna().sum().sum())}, filas duplicadas: {int(df.duplicated().sum())}")
    print(f"Exactitud de declarar a todos sin fraude: {1 - y.mean():.2%}")
    hora = pd.to_datetime(df.fecha_registro).dt.strftime("%H:%M:%S")
    print(f"fecha_registro: {hora.nunique()} hora(s) del día distinta(s): {hora.unique().tolist()}")
    print(f"ROC-AUC de alerta_sistema sola: {roc_auc_score(y, df.alerta_sistema):.3f}")
    r2 = LinearRegression().fit(df[SEGURIDAD], df.score_riesgo).score(df[SEGURIDAD], df.score_riesgo)
    print(f"R² de score_riesgo explicado por los 10 indicadores de seguridad: {r2:.3f}")
    print(f"monto_operacion: mediana S/ {df.monto_operacion.median():.0f}, máximo S/ {df.monto_operacion.max():.0f}")
    print(f"ingreso_mensual: mediana S/ {df.ingreso_mensual.median():.0f}")
    frec = df.tipo_operacion_frecuente.value_counts()
    print(f"tipo_operacion_frecuente: {len(frec)} categorías, de {frec.min()} a {frec.max()} clientes")

    titulo("Hipótesis (Experimentos): tasa de fraude en % según el valor de la variable")
    for col in ["ubicacion_inusual", "dispositivo_nuevo", "intentos_contacto_sospechoso", "tiene_token_digital"]:
        print(f"{col}: {tasa(df, col)}")
    planas = pd.concat([df.groupby(c)[C.TARGET].mean() for c in ["departamento", "banco_principal", "app_preferida"]])
    print(f"departamento, banco_principal y app_preferida: tasas entre {planas.min():.1%} y {planas.max():.1%}")
    n_planas = sum(df[c].nunique() for c in ["departamento", "banco_principal", "app_preferida"])
    print(f"Columnas one-hot de esas 3 variables: {n_planas}")

    titulo("Condiciones experimentales (Tabla I)")
    folds = list(particiones().split(X, y))
    tr_n = sorted({len(tr) for tr, _ in folds}); te_n = sorted({len(te) for _, te in folds})
    tr_f = sorted({int(y[tr].sum()) for tr, _ in folds}); te_f = sorted({int(y[te].sum()) for _, te in folds})
    tasas = [y[te].mean() for _, te in folds]
    print(f"Folds: {len(folds)}. Entrenamiento: {tr_n} clientes, {tr_f} fraudes. Prueba: {te_n} clientes, {te_f} fraudes")
    print(f"Proporción de fraudes en los folds de prueba: de {min(tasas):.2%} a {max(tasas):.2%}")
    tr, te = folds[0]
    prep = construir_preprocesamiento(X).fit(X.iloc[tr])
    print(f"Variables después del preprocesamiento: {prep.transform(X.iloc[te]).shape[1]}")
    modelo, usa = obtener_modelos()["logreg"]
    bb = construir_pipeline(modelo, X, usa).fit(X.iloc[tr], y[tr]).named_steps["modelo"]
    sub = {(len(e.named_steps["sampler"].sample_indices_),
            int(y[tr][e.named_steps["sampler"].sample_indices_].sum())) for e in bb.estimators_}
    print(f"Fold 1: {len(bb.estimators_)} submuestras de (filas, fraudes) = {sub}")
    print(f"Tamaño de submuestra en los 25 folds (2 x fraudes del entrenamiento): {sorted({2 * f for f in tr_f})}")

    titulo("Resultados: comparación global")
    prev = C.N_FRAUDES_ESPERADOS / C.N_FILAS_ESPERADAS
    for m in MODELOS:
        d = metricas[m].pr_auc
        print(f"{m:8s} PR-AUC media / prevalencia = {d.mean() / prev:.1f} veces; rango por fold {d.min():.3f}–{d.max():.3f}")
    piv = pd.concat(metricas.values()).pivot(index="fold", columns="modelo", values="pr_auc")
    for m in ["arbol", "knn", "xgboost"]:
        print(f"Folds en que logreg supera a {m}: {int((piv.logreg > piv[m]).sum())} de {len(piv)}")
    print(f"Diferencia de PR-AUC media XGBoost - KNN: {piv.xgboost.mean() - piv.knn.mean():.3f}")
    print(f"DE del recall top 5 % de XGBoost: {metricas['xgboost'].recall_top5.std():.3f}")

    titulo("Resultados: lectura operativa (Fig. 3)")
    d = pred["logreg"][pred["logreg"].fold == 0].sort_values("prob", ascending=False, kind="stable")
    print(f"Fold 1, regresión logística: {int(d.y_real.head(60).sum())} de {int(d.y_real.sum())} fraudes "
          f"en los 60 primeros; {int(d.y_real.tail(600).sum())} en los 600 últimos")
    medias = {}
    for m in MODELOS:
        curvas, marcas = curvas_captura(pred[m])
        i5 = np.argmin(abs(REJILLA - 0.05))
        medias[m] = curvas.mean(axis=0)
        print(f"{m:8s} captura al 5 %: {marcas[0.05]:.1%}, al 20 %: {marcas[0.2]:.1%}; "
              f"banda p10–p90 al 5 %: {np.percentile(curvas[:, i5], 10):.0%}–{np.percentile(curvas[:, i5], 90):.0%}")
    mas_alla = REJILLA >= 0.20
    dif = max(abs(medias[a][mas_alla] - medias[b][mas_alla]).max()
              for a, b in [("logreg", "knn"), ("logreg", "xgboost"), ("knn", "xgboost")])
    print(f"Máxima diferencia entre las curvas promedio de E1, E3 y E4 desde el 20 %: {dif:.1%}")

    titulo("Resultados: contraste de E2 (árbol)")
    g = pred["arbol"].groupby("fold")
    empates = [int(np.isclose(f.prob, np.sort(f.prob.to_numpy())[::-1][59]).sum()) for _, f in g]
    print(f"Valores distintos de puntaje por fold: {g.prob.nunique().mean():.1f} en promedio")
    print(f"Clientes empatados con el puesto 60: {np.mean(empates):.1f} en promedio")
    for m in ["arbol", "logreg"]:
        r1 = pred[m][pred[m].fold // C.N_SPLITS == 0]
        precision, recall, _ = precision_recall_curve(r1.y_real, r1.prob)
        sel = (recall >= 0.05) & (recall <= 0.60)
        print(f"{m:8s} repetición 1: precisión entre {precision[sel].min():.2f} y {precision[sel].max():.2f} "
              f"para recall de 0,05 a 0,60")


if __name__ == "__main__":
    main()
