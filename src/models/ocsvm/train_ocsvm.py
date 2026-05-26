#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_ocsvm.py
==============

Treina um modelo One-Class SVM usando apenas as amostras da
**elipse predominante** (ou seja, o dataset gerado por `train_pca.py`).

O modelo e o *scaler* são gravados em:
    src/models/ocsvm/ocsvm_model.pkl
    src/models/ocsvm/ocsvm_scaler.pkl
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# 1 - CAMINHOS RELATIVOS
ROOT_DIR   = Path(__file__).resolve().parents[3]  # project_root
METRICS    = ROOT_DIR / "reports" / "metrics"
OCSVM_DIR  = ROOT_DIR / "src" / "models" / "ocsvm"
OCSVM_DIR.mkdir(parents=True, exist_ok=True)

# O dataset da elipse predominante tem nome padrão:
#   mrc_total_curated_data_PCA_PREDOMINANTE_<FORMULACAO>_<TAG>.csv
pca_dataset_path = next(METRICS.glob("*PREDOMINANTE*.csv"))

print(f"Dataset encontrado: {pca_dataset_path.name}")

# 2 - LEITURA DO DATASET
df = pd.read_csv(pca_dataset_path)

# Selecionar apenas as colunas de métrica que serão usadas no modelo.
# Neste exemplo, são as mesmas que foram usadas no treinamento original.
features = ["D_centro", "G_centro", "G_fwhm"]
X = df[features].values

# 3 - PADRONIZAÇÃO (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4 - TREINO DO ONE‑CLASS SVM
model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)  # nu = taxa esperada de outliers
model.fit(X_scaled)

# 5 - SERIALIZAÇÃO
joblib.dump(model, OCSVM_DIR / "ocsvm_model.pkl")
joblib.dump(scaler, OCSVM_DIR / "ocsvm_scaler.pkl")

print(f"\u2705 Modelo OCSVM salvo em: {OCSVM_DIR / 'ocsvm_model.pkl'}")
print(f"\u2705 Scaler salvo em:      {OCSVM_DIR / 'ocsvm_scaler.pkl'}")
