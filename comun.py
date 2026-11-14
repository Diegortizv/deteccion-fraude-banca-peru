"""
Funciones compartidas: carga de datos, pipeline, particiones y métricas.
NO MODIFICAR: garantizan que todos los experimentos usen el mismo proceso.
"""
import hashlib

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config as C


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_datos():
    """Carga el Excel, verifica que sea el archivo correcto y separa X / y."""
    if not C.DATA_PATH.exists():
        raise FileNotFoundError(f"No encuentro {C.DATA_PATH}. Descarga el Excel de Kaggle "
                                "y colócalo en la carpeta datos/ con ese nombre.")
    huella = sha256_archivo(C.DATA_PATH)
    if huella != C.DATA_SHA256:
        raise ValueError("El Excel no coincide con el del equipo (SHA-256 distinto).\n"
                         f"  esperado: {C.DATA_SHA256}\n  obtenido: {huella}")
    df = pd.read_excel(C.DATA_PATH)
    y = df[C.TARGET].to_numpy(dtype=int)
    X = df.drop(columns=[C.TARGET] + C.EXCLUIR)
    assert len(df) == C.N_FILAS_ESPERADAS, f"Se esperaban {C.N_FILAS_ESPERADAS} filas"
    assert y.sum() == C.N_FRAUDES_ESPERADOS, f"Se esperaban {C.N_FRAUDES_ESPERADOS} fraudes"
    return X, y


def construir_preprocesamiento(X):
    """log1p + escalado a montos e ingreso, escalado al resto de numéricas, one-hot a categóricas."""
    numericas = [c for c in X.columns if c not in C.CATEGORICAS]
    otras = [c for c in numericas if c not in C.LOG1P]
    return ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), C.LOG1P),
        ("num", StandardScaler(), otras),
        ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAS),
    ])


def construir_pipeline(modelo, X, usa_ensemble=True):
    """
    Preprocesamiento + (ensemble balanceado de) modelo base.
    El ensemble arma K_SUBMODELOS submuestras 1:1 con todos los fraudes del train fold
    y no-fraudes al azar (EasyEnsemble), entrena un modelo por submuestra y promedia
    las probabilidades. Todo se ajusta solo con el train fold.
    """
    modelo = clone(modelo)
    if usa_ensemble:
        modelo = BalancedBaggingClassifier(
            estimator=modelo,
            n_estimators=C.K_SUBMODELOS,
            sampling_strategy=C.SAMPLING_STRATEGY,
            replacement=False,
            bootstrap=False,
            random_state=C.SEED,
            n_jobs=-1,
        )
    return Pipeline([("prep", construir_preprocesamiento(X)), ("modelo", modelo)])


def particiones():
    """Los mismos 25 folds para todos (5 folds estratificados x 5 repeticiones)."""
    return RepeatedStratifiedKFold(n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS,
                                   random_state=C.SEED)


def metricas_top_k(y_true, score, k=None):
    """Recall y precisión al revisar el top k% de clientes con mayor riesgo."""
    k = C.TOP_K if k is None else k
    n = int(np.ceil(k * len(score)))
    orden = np.argsort(-score, kind="stable")[:n]
    aciertos = int(y_true[orden].sum())
    return aciertos / y_true.sum(), aciertos / n


def evaluar(nombre, modelo, usa_ensemble, X, y, verbose=True):
    """Corre los 25 folds. Devuelve (métricas por fold, predicciones de cada test fold)."""
    filas, predicciones = [], []
    for fold, (tr, te) in enumerate(particiones().split(X, y)):
        pipe = construir_pipeline(modelo, X, usa_ensemble)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        recall_k, precision_k = metricas_top_k(y[te], p)
        filas.append(dict(
            modelo=nombre, fold=fold, repeticion=fold // C.N_SPLITS,
            pr_auc=average_precision_score(y[te], p),
            roc_auc=roc_auc_score(y[te], p),
            recall_top5=recall_k, precision_top5=precision_k,
            n_test=len(te), fraudes_test=int(y[te].sum()),
        ))
        predicciones.append(pd.DataFrame(dict(modelo=nombre, fold=fold, indice=te,
                                              y_real=y[te], prob=p)))
        if verbose:
            print(f"  [{nombre}] fold {fold + 1:2d}/{C.N_SPLITS * C.N_REPEATS}  "
                  f"PR-AUC={filas[-1]['pr_auc']:.3f}")
    return pd.DataFrame(filas), pd.concat(predicciones, ignore_index=True)
