#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py
=======

Orquestra a execução sequencial dos três passos do pipeline:

1️⃣ **PCA** (interativo) – permite ao usuário escolher as colunas a usar.
2️⃣ **Treino do One‑Class SVM** – utiliza o dataset gerado pela elipse predominante.
3️⃣ **Predição** – classifica um arquivo de teste (padrão: test/negative_test.csv).

Uso:
    python main.py            # segue fluxo completo; o passo 1 solicitará a escolha das colunas.
    python main.py --no‑pca   # pula a etapa interativa (usa o último PCA já salvo).
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------
def run_module(module_path: Path):
    """
    Executa um módulo Python como subprocesso.

    Mostra stdout/stderr em tempo real no terminal,
    permitindo interação via input().
    """
    cmd = [sys.executable, str(module_path)]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n⚠️ Erro ao executar: {module_path.name}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Pipeline completo (PCA → OCSVM → predict)")
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Pular a fase interativa de PCA (usa o último modelo já salvo).",
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # 1 – PCA (interativo)
    # ------------------------------------------------------------------
    if not args.no_pca:
        print("\n=== Etapa 1 – Treino PCA (interativo) ===")
        pca_script = ROOT / "src" / "models" / "pca" / "train_pca.py"
        run_module(pca_script)
    else:
        print("\n=== Etapa 1 – Pulada (pelo flag --no-pca) ===")

    # ------------------------------------------------------------------
    # 2 – Treino One‑Class SVM
    # ------------------------------------------------------------------
    print("\n=== Etapa 2 – Treino One‑Class SVM ===")
    ocsvm_script = ROOT / "src" / "models" / "ocsvm" / "train_ocsvm.py"
    run_module(ocsvm_script)

    # ------------------------------------------------------------------
    # 3 – Predição de amostra de teste
    # ------------------------------------------------------------------
    print("\n=== Etapa 3 – Predição de amostra de teste ===")
    predict_script = ROOT / "services" / "predict_ocsvm.py"
    run_module(predict_script)

    print("\nPipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
