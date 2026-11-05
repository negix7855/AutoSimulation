import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

import active_sim_bgr as al
from bgr import LABEL, STUDY_DIR, RESULTS_CSV


# ============================================================
# BGR の ok を、bgr.py の全探索の結果 (bgr_results.csv) すべてで MLP を 1 回学習して予測する
# (5tota/classify_5tota_wsplit.py の BGR 版。能動学習の結果と比べる基準)
#
#   入力 : W, L, R1, R2, R3, m1, m2, m3 (すべて log。active_sim_bgr.to_features と同じ)
#   出力 : ok
#
#   モデル (どちらも 5tota と同じ MLP (64, 64, 32)):
#     MLP          : そのまま学習 (5tota と同じ)
#     MLP_balanced : ok の点を繰り返して ok / not ok の数を揃えて学習
#                    (active_sim_bgr.py の能動学習と同じ。ok が 0.5% しかないため)
#
#   評価 (全探索は全部学習に使うので、学習に使っていない点で評価する):
#     random : bgr_active_sim_random/samples.csv (設計空間から一様に選んだ点。偏りが無い)
#     active : bgr_active_sim/samples.csv (能動学習で選んだ点。境界付近が多く厳しい)
#     grid   : 全探索そのもの (学習データへの当てはまり。参考)
#   能動学習の最後のモデル (bgr_active_sim/models/ensemble.joblib) も同じ点で評価して並べる。
#   ただし能動学習のモデルは active の点で学習しているので、active での成績は参考。
#
#   出力 (study/bgr_classifier/):
#     metrics.csv    : モデル × 評価データごとの指標 (しきい値 0.5)
#     thresholds.csv : しきい値を変えたときの F1 (random の点)
#     pr_curves.png  : random / active の点での PR 曲線
#     models/*.joblib
#
#   Windows 側で実行する想定 (図に日本語を使う。ngspice は使わない)。
# ============================================================


# ============================================================
# パス・設定
# ============================================================

OUT_DIR = os.path.join(STUDY_DIR, "bgr_classifier")
MODEL_DIR = os.path.join(OUT_DIR, "models")

EVAL_SETS = {
    "random": os.path.join(STUDY_DIR, "bgr_active_sim_random", "samples.csv"),
    "active": os.path.join(STUDY_DIR, "bgr_active_sim", "samples.csv"),
}

AL_MODEL = os.path.join(STUDY_DIR, "bgr_active_sim", "models", "ensemble.joblib")

SEED = 0

THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9]


# ============================================================
# データ
# ============================================================

def load_xy(path):
    df = pd.read_csv(path, usecols=al.PARAM_KEYS + [LABEL])
    df = df.dropna()
    return al.to_features(df[al.PARAM_KEYS].to_numpy()), df[LABEL].to_numpy().astype(int)


# ============================================================
# モデル
# ============================================================

def build_mlp():
    """
    5tota/classify_5tota_wsplit.py の MLP と同じ構成
    """

    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 64, 32),
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=SEED,
            verbose=True,
        ),
    )


def evaluate(y_true, prob, th=0.5):
    """
    しきい値 th での指標と、しきい値によらない指標
    """

    pred = (prob >= th).astype(int)

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
    }


# ============================================================
# 図
# ============================================================

def plot_pr_curves(evals, probs, path):
    """
    評価データごとに、各モデルの PR 曲線
    """

    names = [n for n in EVAL_SETS if n in evals]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5),
                             squeeze=False)

    for ax, ev in zip(axes[0], names):
        y = evals[ev][1]
        for model, p in probs.items():
            if ev not in p:
                continue
            precision, recall, _ = precision_recall_curve(y, p[ev])
            ap = average_precision_score(y, p[ev])
            ax.plot(recall, precision, lw=1.2, label=f"{model} (PR-AUC {ap:.3f})")

        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"{ev} の点 ({len(y)} 点, ok 率 {y.mean():.4f})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

    X, y = load_xy(RESULTS_CSV)
    print(f"全探索 (学習): {len(y)} 行, ok 率 {y.mean():.4f}")

    evals = {"grid": (X, y)}
    for name, path in EVAL_SETS.items():
        if os.path.exists(path):
            evals[name] = load_xy(path)
            print(f"評価 {name}: {len(evals[name][1])} 行, "
                  f"ok 率 {evals[name][1].mean():.4f}")

    probs = {}
    rows = []
    fit_s = {}

    for name, balanced in [("MLP", False), ("MLP_balanced", True)]:
        Xt, yt = al.balance(X, y, SEED) if balanced else (X, y)
        print(f"\n[{name}] 学習 {len(yt)} 行 ...", flush=True)

        model = build_mlp()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(Xt, yt)
        fit_s[name] = time.perf_counter() - t0
        print(f"  学習 {fit_s[name] / 60:.1f} 分, "
              f"{model[-1].n_iter_} エポック", flush=True)

        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        probs[name] = {ev: model.predict_proba(Xe)[:, 1]
                       for ev, (Xe, _) in evals.items()}

    # 能動学習の最後のモデル (5 個の MLP の平均)
    if os.path.exists(AL_MODEL):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ens = joblib.load(AL_MODEL)
        probs["active_learning"] = {ev: al.predict_ensemble(ens, Xe)
                                    for ev, (Xe, _) in evals.items()}

    for model, p in probs.items():
        for ev, (_, ye) in evals.items():
            rows.append({"model": model, "eval": ev, "n_eval": len(ye),
                         **evaluate(ye, p[ev]),
                         "fit_s": fit_s.get(model, np.nan)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n指標 (しきい値 0.5):")
    print(metrics.drop(columns=["fit_s"]).round(4).to_string(index=False))

    # しきい値を変えたときの F1 (偏りの無い random の点で)
    if "random" in evals:
        yr = evals["random"][1]
        th_rows = [{"model": m, "threshold": th,
                    **evaluate(yr, p["random"], th)}
                   for m, p in probs.items() for th in THRESHOLDS]
        th_df = pd.DataFrame(th_rows)
        th_df.to_csv(os.path.join(OUT_DIR, "thresholds.csv"), index=False)
        print("\nrandom の点で、しきい値を変えたときの F1:")
        print(th_df.pivot(index="model", columns="threshold", values="f1")
              .round(4).to_string())

    plot_pr_curves(evals, probs, os.path.join(OUT_DIR, "pr_curves.png"))

    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
