"""
Los experimentos: lo ÚNICO que cambia entre ellos es el modelo base.

Cada entrada es  nombre -> (modelo, usa_ensemble).
Los hiperparámetros están fijos y definidos de antemano (sin búsqueda).
Ningún modelo usa class_weight ni scale_pos_weight: cada submuestra del ensemble
ya viene balanceada 1:1.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config as C


def obtener_modelos():
    return {
        # E0 - piso de referencia: predice al azar respetando la proporción de clases
        "dummy": (DummyClassifier(strategy="stratified", random_state=C.SEED), False),
        # E1 - caso base
        "logreg": (LogisticRegression(C=1.0, max_iter=2000), True),
        # E2
        "arbol": (DecisionTreeClassifier(random_state=C.SEED), True),
        # E3
        "knn": (KNeighborsClassifier(n_neighbors=15), True),
        # E4 (n_jobs=1 porque el ensemble ya paraleliza)
        "xgboost": (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", n_jobs=1, random_state=C.SEED), True),
    }
