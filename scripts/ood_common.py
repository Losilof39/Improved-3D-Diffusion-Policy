"""Shared helpers for PCA / Mahalanobis OOD analysis scripts (analyze_ood*.py)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca(train_data, n_components=2):
    scaler = StandardScaler().fit(train_data)
    pca = PCA(n_components=n_components).fit(scaler.transform(train_data))
    return scaler, pca


def mahalanobis_scores(train_data, eval_data):
    """Mahalanobis distance of each row to the training distribution."""
    mean = train_data.mean(axis=0)
    cov = np.cov(train_data, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    def dist(x):
        diff = x - mean
        return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))

    return dist(train_data), dist(eval_data)


def mahalanobis_report(train_data, eval_data, name):
    train_dist, eval_dist = mahalanobis_scores(train_data, eval_data)
    p95, p99 = np.percentile(train_dist, [95, 99])
    frac_95 = (eval_dist > p95).mean() * 100
    frac_99 = (eval_dist > p99).mean() * 100
    lines = [
        f"-- {name}: Mahalanobis distance to training distribution --",
        f"  training self-distance: mean={train_dist.mean():.3f}, p95={p95:.3f}, "
        f"p99={p99:.3f}, max={train_dist.max():.3f}",
        f"  eval distance:          mean={eval_dist.mean():.3f}, "
        f"p95={np.percentile(eval_dist, 95):.3f}, max={eval_dist.max():.3f}",
        f"  eval samples beyond training p95: {frac_95:.1f}%",
        f"  eval samples beyond training p99: {frac_99:.1f}%",
    ]
    return "\n".join(lines)


def plot_pca(train_scaled, eval_scaled, title, out_path, eval_color=None,
              eval_cmap_label='eval timestep', connect_line=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(train_scaled[:, 0], train_scaled[:, 1], gridsize=40, cmap='Greys', mincnt=1)
    fig.colorbar(hb, ax=ax, label='training point density')

    if eval_color is None:
        eval_color = np.arange(len(eval_scaled))

    if connect_line:
        ax.plot(eval_scaled[:, 0], eval_scaled[:, 1], color='red', alpha=0.3, linewidth=1, zorder=1)

    sc = ax.scatter(eval_scaled[:, 0], eval_scaled[:, 1], c=eval_color,
                     cmap='plasma', s=12, alpha=0.8, label='eval rollout', zorder=2)
    fig.colorbar(sc, ax=ax, label=eval_cmap_label)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
