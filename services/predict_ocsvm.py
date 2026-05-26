#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict_ocsvm.py
================

Carrega o modelo One-Class SVM (e o scaler) treinado e classifica uma única
amostra de teste.  O script pode ser usado como:

    python -m src.services.predict_ocsvm /caminho/para/arquivo.csv

ou, se o caminho não for passado, usa o arquivo padrão
`test/negative_test.csv`.

Saída no terminal:

* Valor das três métricas (D_centro, G_centro, G_fwhm);
* Predição (1 → “compatível”, -1 → “fora do padrão”);
* Score de similaridade (valor da função `score_samples`).
"""

import sys
from pathlib import Path

import joblib
import pandas as pd


# 1 - LOCALIZAÇÃO DOS ARTEFATOS (model + scaler)
ROOT_DIR   = Path(__file__).resolve().parents[1]    # project_root
OCSVM_DIR  = ROOT_DIR / "src" / "models" / "ocsvm"

model   = joblib.load(OCSVM_DIR / "ocsvm_model.pkl")
scaler  = joblib.load(OCSVM_DIR / "ocsvm_scaler.pkl")

# 2 - INPUT DO USUÁRIO (arquivo CSV com a amostra)
DEFAULT_CSV = ROOT_DIR / "test" / "negative_test.csv"

if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
else:
    csv_path = DEFAULT_CSV

if not csv_path.is_file():
    raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

df_amostra = pd.read_csv(csv_path)


# 3 - VALIDAÇÃO DAS COLUNAS
features = ["D_centro", "G_centro", "G_fwhm"]
missing = [c for c in features if c not in df_amostra.columns]
if missing:
    raise ValueError(f"Colunas ausentes no CSV: {', '.join(missing)}")

# 4 - PREPARAÇÃO DA AMOSTRA (aceita só a primeira linha caso haja mais)
nova_amostra = df_amostra[features].values
if nova_amostra.shape[0] != 1:
    print(f"Aviso: o CSV contém {nova_amostra.shape[0]} linhas. Usando apenas a primeira.")
    nova_amostra = nova_amostra[0:1]


# 5 - ESCALONAMENTO + PREDIÇÃO
nova_amostra_scaled = scaler.transform(nova_amostra)
pred = model.predict(nova_amostra_scaled)          # 1 = inlier, -1 = outlier
score = model.score_samples(nova_amostra_scaled)  # maior → mais parecido

# 6 - SAÍDA FORMATADA
print("\n" + "=" * 50)
print(f" Avaliação da amostra: {csv_path.name}")
print("=" * 50)
print("Parâmetros da amostra:")
print(f"  D_centro : {nova_amostra[0, 0]:.2f}")
print(f"  G_centro : {nova_amostra[0, 1]:.2f}")
print(f"  G_fwhm   : {nova_amostra[0, 2]:.2f}")
print("-" * 50)

if pred[0] == 1:
    print("\u2705 Amostra COMPATÍVEL com GO")
else:
    print("\u274C Amostra FORA do padrão GO")
print(f"Score de similaridade: {score[0]:.4f}")
print("=" * 50)
