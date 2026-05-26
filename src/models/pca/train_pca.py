#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_pca.py
============

Treina um modelo de Análise de Componentes Principais (PCA) sobre espectros Raman
curados e gera:

* Gráfico de dispersão dos dois primeiros componentes (com elipses de confiança);
* Biplot (PC1/PC2 + vetores das features originais);
* Gráficos de loadings (barras para PC1 e PC2);
* Dataset contendo apenas as amostras que caem dentro da elipse predominante
  (salvo em *reports/metrics*).

O script é **interativo**: o usuário escolhe quais colunas numéricas deverão ser
usadas no cálculo do PCA (todas ou um subconjunto).

Requisitos
----------
* pandas, numpy, matplotlib, seaborn, scikit-learn
* O diretório raiz do projeto deve ser o mesmo onde está o arquivo
  ``train_pca.py`` (ou você pode executar com ``python3 src/models/pca/train_pca.py``).

Estrutura de pastas esperada
----------------------------
one-class-svm-model/
├─ data/
│   └─ lake/gold/mrc/mrc_total_curated_data.csv   <- entrada
├─ reports/
│   ├─ figures/                                   <- saída das figuras
│   └─ metrics/                                   <- dataset da elipse predominante
└─ src/
    └─ models/
        └─ train_pca.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse

# 1 - CONFIGURAÇÕES DE CAMINHOS (relativos ao diretório raiz do projeto)
ROOT_DIR    = Path(__file__).resolve().parents[3]          # project_root
DATA_DIR    = ROOT_DIR / "data" / "lake" / "gold" / "mrc"
REPORT_DIR  = ROOT_DIR / "reports"
FIG_DIR     = REPORT_DIR / "figures"
METRICS_DIR = REPORT_DIR / "metrics"

# Garantir que os diretórios de saída existam
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV   = DATA_DIR / "dataset_curated.csv"

# 2 - FUNÇÕES AUXILIARES
def draw_confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Desenha uma elipse de confiança (n_std desvios-padrão) baseada na
    covariância dos pontos (x, y).  Retorna a elipse, o centro, a escala e
    a matriz de covariância (usados depois para teste de inclusão).
    """
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Raio da elipse em coordenadas unitárias (sem rotação)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        fill=True, **kwargs
    )

    # Escala real (desvios-padrão) e translação para o centro dos dados
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x  = np.mean(x)
    mean_y  = np.mean(y)

    transf = (
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg(45)          # rotação de 45° (convenção do script original)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return ellipse, (mean_x, mean_y), (scale_x, scale_y), cov


def point_in_ellipse(point, center, scale_x, scale_y, cov, n_std=2.0):
    """
    Verifica se `point` está dentro da elipse de confiança.
    Usa distância de Mahalanobis (mais robusta que a avaliação
    geométrica direta).  Retorna True/False.
    """
    x, y = point
    cx, cy = center

    # Vetor deslocado para o centro da elipse
    x_t = x - cx
    y_t = y - cy

    try:
        inv_cov = np.linalg.inv(cov)
        vec = np.array([x_t, y_t])
        d = np.sqrt(vec @ inv_cov @ vec)          # distância de Mahalanobis
        # Para 2D normal, a elipse n_std contém pontos com d <= sqrt(2) * n_std
        return d <= np.sqrt(2) * n_std
    except np.linalg.LinAlgError:
        # Covariância singular → consideramos fora
        return False


def add_ellipse_label(ax, center, count, total_points, color, offset=(0, 0)):
    """
    Insere o número de pontos dentro da elipse como label.
    Ajusta cor do texto de acordo com a luminância do fundo.
    """
    x, y = center
    off_x, off_y = offset
    text   = f"{count}"
    perc   = (count / total_points) * 100

    # Luminância para escolher texto branco ou preto
    r, g, b = color[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    txt_color = "white" if lum < 0.5 else "black"

    bbox = dict(
        boxstyle="round,pad=0.3",
        facecolor=color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    ax.text(
        x + off_x,
        y + off_y,
        text,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=bbox,
        color=txt_color,
    )
    return perc


def plot_feature_vectors(ax, pca, features, scale=1.0, colors=None):
    """
    Cria o biplot: desenha setas que representam os loadings das
    features originais no espaço PC1-PC2.
    """
    loadings = pca.components_.T                         # (n_features, n_components)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(features)))

    for i, (feat, vec) in enumerate(zip(features, loadings)):
        x, y = vec[0] * scale, vec[1] * scale
        ax.arrow(
            0, 0, x, y,
            head_width=0.05 * scale,
            head_length=0.05 * scale,
            fc=colors[i],
            ec=colors[i],
            linewidth=2,
            zorder=15,
        )
        # rótulo da feature
        ax.text(
            x * 1.15,
            y * 1.15,
            feat,
            fontsize=10,
            fontweight="bold",
            color=colors[i],
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.7,
                edgecolor=colors[i],
            ),
            zorder=15,
        )

    # círculo de referência (facilita visualização da escala)
    ax.add_patch(plt.Circle((0, 0), scale, fill=False,
                           linestyle="--", linewidth=1, color="gray", alpha=0.5,
                           zorder=10))


def plot_loadings_bar(pca, features, explained_variance):
    """
    Gráficos de barras dos loadings (PC1 e PC2) – útil para inspeção.
    """
    loadings = pca.components_               # (n_components, n_features)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

    colors1 = plt.cm.Reds(np.linspace(0.3, 0.9, len(features)))
    colors2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))

    # PC1
    bars1 = ax1.bar(range(len(features)), loadings[0], color=colors1, edgecolor="darkred")
    ax1.set_xticks(range(len(features)))
    ax1.set_xticklabels(features, rotation=45, ha="right", fontweight="bold")
    ax1.set_ylabel("Loading", fontweight="bold")
    ax1.set_title(f"Loadings PC1 ({explained_variance[0]:.2f}%)", fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars1, loadings[0]):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02 if h > 0 else h - 0.08,
            f"{val:.3f}",
            ha="center",
            va="bottom" if h > 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    # PC2
    bars2 = ax2.bar(range(len(features)), loadings[1], color=colors2, edgecolor="darkblue")
    ax2.set_xticks(range(len(features)))
    ax2.set_xticklabels(features, rotation=45, ha="right", fontweight="bold")
    ax2.set_ylabel("Loading", fontweight="bold")
    ax2.set_title(f"Loadings PC2 ({explained_variance[1]:.2f}%)", fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars2, loadings[1]):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02 if h > 0 else h - 0.08,
            f"{val:.3f}",
            ha="center",
            va="bottom" if h > 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


# 3 - LEITURA DOS DADOS
df = pd.read_csv(INPUT_CSV)

# Cria coluna "prefixo" (agrupamento por formulação)
df["prefixo"] = df["nome_amostra"].apply(lambda x: "_".join(x.split("_")[:-1]))

# 4 - SELEÇÃO INTERATIVA DE VARIÁVEIS NUMÉRICAS
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nColunas numéricas disponíveis para PCA:\n")
for i, col in enumerate(numeric_cols, start=1):
    print(f"{i} - {col}")

print("\nDigite:")
print("  all  → usar todas as variáveis")
print("  números separados por vírgula → escolher variáveis")

user_input = input("\nSeleção: ").strip()

if user_input.lower() == "all":
    features = numeric_cols
else:
    try:
        indices = [int(i.strip()) - 1 for i in user_input.split(",")]
        features = [numeric_cols[i] for i in indices]
    except Exception as exc:
        raise ValueError(
            "Entrada inválida. Use 'all' ou números separados por vírgula."
        ) from exc

print("\nVariáveis usadas no PCA:")
for f in features:
    print(" -", f)

# 5 - NOME DO ARQUIVO DE SAÍDA (baseado nas variáveis escolhidas)
base_name = INPUT_CSV.stem
if len(features) == len(numeric_cols):
    feature_tag = "ALL"
else:
    feature_tag = "_".join(features)
feature_tag = feature_tag.replace("/", "_").replace(" ", "")

pca_png          = FIG_DIR / f"{base_name}_PCA_{feature_tag}.png"
biplot_png       = FIG_DIR / f"{base_name}_PCA_BIPLOT_{feature_tag}.png"
loadings_png     = FIG_DIR / f"{base_name}_PCA_LOADINGS_{feature_tag}.png"

print("\nArquivos de saída serão:")
print(f"  - PCA:          {pca_png}")
print(f"  - Biplot:       {biplot_png}")
print(f"  - Loadings:     {loadings_png}")

# 6 - PREPARAÇÃO DOS DADOS PARA O PCA
X = df[features].values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    principal_components, columns=["PC1", "PC2"]
)
pca_df["prefixo"] = df["prefixo"]
pca_df["nome_amostra"] = df["nome_amostra"]

explained_variance = pca.explained_variance_ratio_ * 100
total_points = len(pca_df)

# 7 - DEBUG / RELATÓRIOS INICIAIS
print("\n" + "=" * 70)
print("🔍 DEBUG - VERIFICAÇÃO DOS GRUPOS")
print("=" * 70)
print(f"\U0001F4CA Total de amostras: {total_points}")
print(f"\U0001F4CA Grupos únicos: {pca_df['prefixo'].nunique()}")

grupos_contagem = pca_df["prefixo"].value_counts().sort_index()
for grp, cnt in grupos_contagem.items():
    print(f"   {grp}: {cnt} amostras")
print("=" * 70 + "\n")

# 8 - PLOT DO PCA COM ELIPSES
sns.set_style("whitegrid")
n_grupos = pca_df["prefixo"].nunique()
palette = sns.color_palette("viridis", n_grupos)

plt.figure(figsize=(8, 8))
ax = sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="prefixo",
    style="prefixo",
    s=100,
    palette=palette,
    alpha=0.8,
    zorder=5,
)

# ----- desenhar elipses e coletar informações -----
ellipse_info = []          # dicionário por elipse
for idx, (prefixo, group) in enumerate(pca_df.groupby("prefixo")):
    ellipse, center, scales, cov = draw_confidence_ellipse(
        group["PC1"], group["PC2"], ax, n_std=2,
        facecolor=palette[idx],
        edgecolor=palette[idx],
        alpha=0.15,
        linewidth=2,
        zorder=1,
    )

    # conta quantos pontos do dataset total caem dentro da elipse
    pts_inside = sum(
        point_in_ellipse(
            (row["PC1"], row["PC2"]), center, scales[0], scales[1], cov, n_std=2
        )
        for _, row in pca_df.iterrows()
    )

    ellipse_info.append({
        "prefixo": prefixo,
        "center": center,
        "points_inside": pts_inside,
        "percentage": pts_inside / total_points * 100,
        "color": palette[idx],
        "group_size": len(group),
    })

# ordenar por número de pontos (decrescente) e pegar a elipse predominante
ellipse_info_sorted = sorted(ellipse_info, key=lambda x: x["points_inside"], reverse=True)
best_ellipse = ellipse_info_sorted[0]

# 9 - GERAÇÃO DO DATASET DA ELIPSE PREDOMINANTE
print("\n" + "=" * 70)
print("GERANDO DATASET DA ELIPSE PREDOMINANTE")
print("=" * 70)

indices_inside = []
# Refaz a elipse para o grupo que é a predominante (mais segura)
group_best = pca_df[pca_df["prefixo"] == best_ellipse["prefixo"]]
cov_best   = np.cov(group_best["PC1"], group_best["PC2"])
center_best = (np.mean(group_best["PC1"]), np.mean(group_best["PC2"]))
scale_x = np.sqrt(cov_best[0, 0]) * 2  # n_std=2
scale_y = np.sqrt(cov_best[1, 1]) * 2

for idx, row in pca_df.iterrows():
    if point_in_ellipse(
        (row["PC1"], row["PC2"]), center_best, scale_x, scale_y, cov_best, n_std=2
    ):
        indices_inside.append(idx)

dataset_predominante = df.loc[indices_inside, features].copy()
dataset_predominante["nome_amostra"] = df.loc[indices_inside, "nome_amostra"]
dataset_predominante["prefixo"]      = df.loc[indices_inside, "prefixo"]

# imprimir resumo
print(f"\nFormulação predominante: {best_ellipse['prefixo']}")
print(f"Total de amostras originais: {total_points}")
print(f"Amostras dentro da elipse: {len(dataset_predominante)} "
      f"({len(dataset_predominante) / total_points * 100:.1f}%)")
print("Parâmetros incluídos:", ", ".join(features))

# salvar csvs
formulacao_limpa = best_ellipse["prefixo"].replace("/", "_").replace(" ", "")
output_dataset = METRICS_DIR / f"{base_name}_PCA_PREDOMINANTE_{formulacao_limpa}_{feature_tag}.csv"
output_dataset_completo = METRICS_DIR / f"{base_name}_PCA_PREDOMINANTE_{formulacao_limpa}_COMPLETO.csv"

dataset_predominante.to_csv(output_dataset, index=False)
df.loc[indices_inside].to_csv(output_dataset_completo, index=False)

print(f"Dataset (colunas selecionadas) salvo em: {output_dataset}")
print(f"Dataset completo salvo em: {output_dataset_completo}")
print("=" * 70)

# 10 - ADICIONAR LABEL NUMÉRICO EM CADA ELIPSE
positions = [(0, .15), (.15, 0), (0, -.15), (-.15, 0),
             (.15, .15), (-.15, .15), (.15, -.15), (-.15, -.15)]

for i, info in enumerate(ellipse_info):
    pos = positions[i % len(positions)]
    add_ellipse_label(ax, info["center"], info["points_inside"],
                      total_points, info["color"], offset=pos)

# 11 - DESTACAR ELIPSE PREDOMINANTE
group_best = pca_df[pca_df["prefixo"] == best_ellipse["prefixo"]]
draw_confidence_ellipse(
    group_best["PC1"], group_best["PC2"], ax,
    n_std=2, facecolor="none", edgecolor="red",
    alpha=1.0, linewidth=4, zorder=10,
)

# 12 - ESTÉTICA FINAL DO GRÁFICO
plt.xlabel(f"PC1 ({explained_variance[0]:.2f}%)", fontsize=12, fontweight="bold")
plt.ylabel(f"PC2 ({explained_variance[1]:.2f}%)", fontsize=12, fontweight="bold")
ax.tick_params(axis="both", labelsize=11, width=1.5)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
for spine in ax.spines.values():
    spine.set_linewidth(3.0)

# quadro informativo da elipse predominante
info_box = [
    "Predominância",
    best_ellipse["prefixo"],
    f"{best_ellipse['points_inside']} pts ({best_ellipse['percentage']:.1f}%)",
]
bbox_props = dict(
    boxstyle="round,pad=0.5",
    facecolor="white",
    alpha=0.9,
    edgecolor="gray",
    linewidth=1.5,
)
plt.text(
    0.02,
    0.98,
    "\n".join(info_box),
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=bbox_props,
    family="sans-serif",
)
plt.legend(title="Prefixo da Amostra", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(pca_png, dpi=300, bbox_inches="tight")
plt.close()

# 13 - BIPLOT (PCA + vetores das features)
print("\nGERANDO BIPLOT COM VETORES DAS FEATURES")
plt.figure(figsize=(10, 6))
ax2 = sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="prefixo",
    style="prefixo",
    s=100,
    palette=palette,
    alpha=0.6,
    zorder=5,
)

# elipses transparentes (para o biplot)
for idx, (prefixo, group) in enumerate(pca_df.groupby("prefixo")):
    draw_confidence_ellipse(
        group["PC1"], group["PC2"], ax2, n_std=2,
        facecolor=palette[idx],
        edgecolor=palette[idx],
        alpha=0.08,
        linewidth=1.5,
        zorder=1,
    )

# destacar a elipse predominante
draw_confidence_ellipse(
    group_best["PC1"], group_best["PC2"], ax2,
    n_std=2, facecolor="none", edgecolor="red",
    alpha=1.0, linewidth=3, zorder=10,
)

# vetores das features (biplot)
max_coord = np.abs(pca_df[["PC1", "PC2"]].values).max()
vector_scale = max_coord * 0.8
plot_feature_vectors(ax2, pca, features, scale=vector_scale)

# estética extra (bordas, ticks em negrito, legenda)
for spine in ax2.spines.values():
    spine.set_linewidth(3.0)
    spine.set_color("black")
ax2.tick_params(axis="both", labelsize=11, width=1.5, direction="out")
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight("bold")
legend = ax2.legend(title="Prefixo da Amostra", bbox_to_anchor=(1.02, 1), loc="upper left")
legend.get_title().set_fontweight("bold")
for txt in legend.get_texts():
    txt.set_fontweight("bold")

ax2.set_xlabel(f"PC1 ({explained_variance[0]:.2f}%)", fontsize=14, fontweight="bold")
ax2.set_ylabel(f"PC2 ({explained_variance[1]:.2f}%)", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.8, linestyle="-", linewidth=0.6)
ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
ax2.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

# expandir limites para que os vetores caibam
xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
ax2.set_xlim(min(xlim[0], -vector_scale * 1.2), max(xlim[1], vector_scale * 1.2))
ax2.set_ylim(min(ylim[0], -vector_scale * 1.2), max(ylim[1], vector_scale * 1.2))

plt.tight_layout()
plt.savefig(biplot_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"\u2705 Biplot salvo em: {biplot_png}")

# ----------------------------------------------------------------------
# 14 – LOADINGS (barras)
# ----------------------------------------------------------------------
fig_load = plot_loadings_bar(pca, features, explained_variance)
fig_load.savefig(loadings_png, dpi=300, bbox_inches="tight")
plt.close(fig_load)
print(f"\u2705 Loadings salvo em: {loadings_png}")
