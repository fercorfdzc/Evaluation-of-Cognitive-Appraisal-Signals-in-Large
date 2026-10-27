"""
validate_stimuli.py
===================
Valida que los estímulos del experimento estén semánticamente alineados.

Salidas:
  - similarity_heatmap.png  -> matriz de calor de similitud coseno (30x30)
  - pca_stimuli.png         -> proyección 2D PCA con colores por grupo
  - similarity_report.txt   -> métricas numéricas
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ── Estilo limpio ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        False,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Estímulos ──────────────────────────────────────────────────

ANXIETY = [
    "I am very nervous about the outcome.",
    "I feel anxious about how this will turn out.",
    "I am worried about the final result.",
    "I feel uneasy regarding the outcome.",
    "I am concerned about what may happen.",
    "The result of this task makes me anxious.",
    "I am feeling nervous about this situation.",
    "I worry that the outcome may not be favorable.",
    "I feel tension regarding the final result.",
    "I am apprehensive about the outcome.",
]

IMPORTANCE = [
    "This task is very important.",
    "This task carries substantial importance.",
    "This problem is highly important.",
    "The successful completion of this task is important.",
    "This question has considerable importance.",
    "The outcome of this task is important.",
    "This is an important task.",
    "The completion of this task is highly important.",
    "This problem has significant importance.",
    "The result of this task is important.",
]

CONSEQUENCES = [
    "There will be significant consequences if the answer is incorrect.",
    "An incorrect answer may lead to important consequences.",
    "Errors in this task may have serious consequences.",
    "The consequences of an incorrect answer are significant.",
    "A wrong answer could result in important consequences.",
    "Incorrect responses may have substantial consequences.",
    "There are meaningful consequences associated with an incorrect answer.",
    "Mistakes in this task may produce important consequences.",
    "An inaccurate answer may have significant consequences.",
    "The impact of an incorrect answer could be substantial.",
]

GROUPS = {
    "Anxiety":      ANXIETY,
    "Importance":   IMPORTANCE,
    "Consequences": CONSEQUENCES,
}

COLORS = {
    "Anxiety":      "#C0392B",
    "Importance":   "#2471A3",
    "Consequences": "#1E8449",
}

# ── Embeddings ─────────────────────────────────────────────────

print("Cargando modelo BAAI/bge-large-en-v1.5 ...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

all_sentences, all_labels = [], []
for gname, sents in GROUPS.items():
    all_sentences.extend(sents)
    all_labels.extend([gname] * len(sents))

print(f"Codificando {len(all_sentences)} frases ...")
embeddings = model.encode(all_sentences, show_progress_bar=True)
sim_matrix  = cosine_similarity(embeddings)
group_names = list(GROUPS.keys())
n = 10

# ══════════════════════════════════════════════════════════════
# FIGURA 1 — Heatmap limpio
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 5.2))

im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0,
               aspect="equal", interpolation="nearest")

# Líneas divisorias
for k in range(1, len(group_names)):
    pos = k * n - 0.5
    ax.axhline(pos, color="white", linewidth=2.5)
    ax.axvline(pos, color="white", linewidth=2.5)

# Etiquetas de grupo
tick_pos = [k * n + n // 2 for k in range(len(group_names))]
ax.set_xticks(tick_pos)
ax.set_yticks(tick_pos)
ax.set_xticklabels(group_names, fontsize=11)
ax.set_yticklabels(group_names, fontsize=11)
ax.tick_params(length=0)

# Borde de bloque diagonal coloreado
for k, gname in enumerate(group_names):
    rect = mpatches.FancyBboxPatch(
        (k*n - 0.5, k*n - 0.5), n, n,
        boxstyle="square,pad=0",
        linewidth=2.0,
        edgecolor=COLORS[gname],
        facecolor="none",
        zorder=3,
    )
    ax.add_patch(rect)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.88)
cbar.set_label("Cosine similarity", fontsize=10)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_linewidth(0.5)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title("Pairwise cosine similarity — BAAI/bge-large-en-v1.5",
             fontsize=11, pad=10)

plt.tight_layout()
plt.savefig("similarity_heatmap.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()
print("[OK] Guardado: similarity_heatmap.png")

# ══════════════════════════════════════════════════════════════
# FIGURA 2 — PCA con elipses de confianza
# ══════════════════════════════════════════════════════════════

def confidence_ellipse(x, y, ax, n_std=1.96, **kw):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=(x.mean(), y.mean()),
                         width=w, height=h, angle=angle, **kw))

pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
var    = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(6, 5))

for gname in group_names:
    mask = np.array(all_labels) == gname
    x, y = coords[mask, 0], coords[mask, 1]
    col  = COLORS[gname]

    # Elipse de confianza 95 %
    confidence_ellipse(x, y, ax, n_std=1.96,
                       facecolor=col, alpha=0.12,
                       edgecolor=col, linewidth=1.2,
                       linestyle="--", zorder=1)

    # Puntos
    ax.scatter(x, y, c=col, label=gname,
               s=55, alpha=0.88,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Centroide
    ax.scatter(x.mean(), y.mean(),
               c=col, s=160, marker="D",
               edgecolors="white", linewidths=0.8, zorder=5)

ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% variance)", labelpad=5)
ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% variance)", labelpad=5)
ax.set_title("Stimulus embeddings — PCA projection (95% CI ellipses)",
             fontsize=11, pad=10)

ax.grid(True, linestyle=":", linewidth=0.5, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)

legend = ax.legend(frameon=True, framealpha=0.9,
                   edgecolor="#CCCCCC", fancybox=False,
                   fontsize=9, loc="best")
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.savefig("pca_stimuli.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()
print("[OK] Guardado: pca_stimuli.png")

# ══════════════════════════════════════════════════════════════
# REPORTE NUMÉRICO
# ══════════════════════════════════════════════════════════════

internal_sims = {}
for i, gname in enumerate(group_names):
    block = sim_matrix[i*n:(i+1)*n, i*n:(i+1)*n]
    internal_sims[gname] = block[~np.eye(n, dtype=bool)].mean()

report_lines = [
    "=" * 60,
    "REPORTE DE VALIDACION SEMANTICA DE ESTIMULOS",
    "Modelo: BAAI/bge-large-en-v1.5  |  Metrica: similitud coseno",
    "=" * 60,
    "\n--- COHESION INTERNA (mayor = mejor) ---",
]
for gname in group_names:
    report_lines.append(f"  {gname:<26} {internal_sims[gname]:.4f}")

report_lines.append("\n--- SEPARACION ENTRE GRUPOS (menor = mejor) ---")
cross_sims = {}
for i, g1 in enumerate(group_names):
    for j, g2 in enumerate(group_names):
        if j <= i:
            continue
        v = sim_matrix[i*n:(i+1)*n, j*n:(j+1)*n].mean()
        cross_sims[(g1, g2)] = v
        report_lines.append(f"  {g1:<26} vs {g2:<26} {v:.4f}")

report_lines.append("\n--- RATIO COHESION / SEPARACION ---")
for (g1, g2), cross in cross_sims.items():
    ratio = (internal_sims[g1] + internal_sims[g2]) / 2 / cross
    report_lines.append(f"  {g1} vs {g2}: {ratio:.3f}x")

report_lines.append("\n" + "=" * 60)
report_text = "\n".join(report_lines)
print("\n" + report_text)

with open("similarity_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("[OK] Guardado: similarity_report.txt")
