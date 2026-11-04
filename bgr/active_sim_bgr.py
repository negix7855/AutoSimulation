import os
import csv
import time
import shutil
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed, dump

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning

import bgr
from bgr import (
    export_netlist,
    FIELDNAMES,
    LABEL,
    SCH_PATH,
    NETLIST_DIR,
    STUDY_DIR,
    RESULTS_CSV,
)


# ============================================================
# バンドギャップリファレンス (BGR_al.sch) の「動作する範囲」(ok) の境界を、
# ngspice を実際に回しながら能動学習で求める
#
#   cascade/active_sim_cascade.py と同じ方法 (MLP 5 個のアンサンブル + エントロピー + k-means)。
#   シミュレーションと判定は bgr.py の simulate_point を使う。
#
#   入力: W, L (MOS 共通), R1, R2, R3, m1, m2, m3 の 8 次元
#   範囲 (PARAM_SPACE) は、bgr.py の全探索 (GRID) の端から余裕をとる
#   正解: ok (0〜100 ℃ で vout の変動が ±2.5 mV 以内、など。bgr.py を参照)
#
#   カスコードとの違い:
#     ok = 1 の点は設計空間のごく一部 (R3 / (R2 − R1) × ln(m2 / m1) がほぼ決まった値の
#     薄い層) にしかない。そのため
#       - change_rate は「前ラウンドと今ラウンドのどちらかで ok と予測された点」に
#         対する、予測が変わった点の割合にする。全点で割ると、ok の点が少ないだけで
#         小さくなり、境界が定まる前に止まってしまう
#       - 評価は accuracy だけでなく F1 / recall / precision も記録する
#         (全部 0 と予測しても accuracy は高くなるため)
#       - 学習時に ok の点を繰り返して ok / not ok の数を揃える (balance)。
#         揃えないと MLP が「すべて 0」と予測し、境界の点を選べない
#
#   補助的な指標 (停止には使わない):
#     est_err  : 直近 STOP_WINDOW ラウンドの「ランダムに選んだ点」で、
#                シミュレーション前の予測が外れていた割合
#     test_*   : bgr.py の全探索の結果 bgr_results.csv での成績 (TEST_SAMPLES 行に間引く)
#                (学習には使わない。AL_TEST=0 で評価しない)
#
#   途中で止めても、もう一度実行すれば続きから再開する。
#
#   出力 (study/bgr_active_sim/。random なら _random、AL_SEED が 0 以外なら _seed<N> を付ける):
#     samples.csv, failed.csv, rounds.csv, rounds.png, stop_pred/,
#     models/ensemble.joblib (cascade と同じ構成)
#
#   環境変数:
#     AL_TARGET   : 停止する change_rate のしきい値 (既定 0.01)
#     AL_BUDGET   : シミュレーション回数の上限
#     AL_TEST     : 0 にすると全探索の結果での評価をしない
#     AL_SEED     : 乱数の種 (既定 0)
#     AL_STRATEGY : active (既定) / random
#     AL_STOP_F1  : 全探索の結果での F1 がこの値以上になったら止める (比較実験用)
#     AL_GROWTH   : 1ラウンドの追加点数の増え方 (既定 0.1。random の既定 0.2)
#     NUM_WORKERS : ngspice の並列数 (既定は CPU コア数)
#     WORK_DIR    : 作業用ネットリストを置く親フォルダ
# ============================================================


# ============================================================
# パス
# ============================================================

SEED = int(os.environ.get("AL_SEED", 0))

STRATEGY = os.environ.get("AL_STRATEGY", "active")
if STRATEGY not in ("active", "random"):
    raise ValueError(f"AL_STRATEGY は active か random: {STRATEGY}")

OUT_DIR = os.path.join(
    STUDY_DIR,
    "bgr_active_sim"
    + ("_random" if STRATEGY == "random" else "")
    + (f"_seed{SEED}" if SEED else "")
)

SAMPLES_PATH = os.path.join(OUT_DIR, "samples.csv")
FAILED_PATH = os.path.join(OUT_DIR, "failed.csv")
ROUNDS_PATH = os.path.join(OUT_DIR, "rounds.csv")
STOP_PRED_DIR = os.path.join(OUT_DIR, "stop_pred")
MODEL_DIR = os.path.join(OUT_DIR, "models")

# 評価用の全探索の結果
TEST_CSV = RESULTS_CSV


# ============================================================
# 設定
# ============================================================

# 振る範囲: (下限, 上限, 種類, 丸める小数点以下の桁数)
#   種類 "log": log で一様 / "int": log で一様な整数
PARAM_SPACE = {
    "W_um": (2.5, 100.0, "log", 2),       # GRID 5.6–44.8
    "L_um": (0.28, 5.0, "log", 2),        # GRID 0.28–2.24 (0.28 未満はモデルの範囲外)
    "R1_ohm": (1.0, 2000.0, "log", 1),    # GRID 1–1000
    "R2_ohm": (20.0, 100000.0, "log", 0),  # GRID 50–50000
    "R3_ohm": (100.0, 500000.0, "log", 0),  # GRID 200–200000
    "m1": (1, 8, "int", 0),               # GRID 1–8
    "m2": (1, 32, "int", 0),              # GRID 2–16
    "m3": (1, 8, "int", 0),               # GRID 1–8
}

PARAM_KEYS = list(PARAM_SPACE)
assert PARAM_KEYS == bgr.PARAM_KEYS, "bgr.GRID と列の順番をそろえる"

SAMPLE_FIELDS = FIELDNAMES + ["round", "role", "pred_before"]

# 最初にランダムに回す点数
N_INIT = 3000

# 1ラウンドで追加する点数 = max(BATCH, それまでの点数 × BATCH_GROWTH)
BATCH = 1000
BATCH_GROWTH = float(os.environ.get(
    "AL_GROWTH", 0.2 if STRATEGY == "random" else 0.1
))

BUDGET = int(os.environ.get(
    "AL_BUDGET", 300_000 if STRATEGY == "random" else 100000
))

_stop_f1 = os.environ.get("AL_STOP_F1", "")
STOP_F1 = float(_stop_f1) if _stop_f1 else None

# 停止条件: change_rate が TARGET_ERR を STOP_PATIENCE ラウンド続けて下回ったら止める
TARGET_ERR = float(os.environ.get("AL_TARGET", 0.01))
STOP_PATIENCE = 3

N_STOP = 100_000
STOP_WINDOW = 3
ENSEMBLE_SIZE = 5
RANDOM_FRAC = 0.2
N_CANDIDATES = 500_000
TOP_FACTOR = 10
PREDICT_CHUNK = 500_000

USE_TEST = os.environ.get("AL_TEST", "1") != "0"
TEST_SAMPLES = 1_000_000

# 評価用データの間引き方は種によらず固定する (種どうしで同じ行で比べるため)
TEST_SEED = 0


# ============================================================
# 設計空間
# ============================================================

def sample_space(rng, n):
    """
    PARAM_SPACE の範囲から n 点を作る (列は PARAM_KEYS の順)
    """

    u = rng.random((n, len(PARAM_KEYS)))
    P = np.empty_like(u)

    for j, k in enumerate(PARAM_KEYS):
        lo, hi, kind, digits = PARAM_SPACE[k]
        if kind == "int":
            # [lo, hi + 1) を log で一様に作って切り捨てる
            P[:, j] = np.floor(
                np.exp(np.log(lo) + u[:, j] * (np.log(hi + 1) - np.log(lo)))
            )
            P[:, j] = np.clip(P[:, j], lo, hi)
        else:
            P[:, j] = np.round(
                np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))),
                digits
            )

    return P


def to_features(P):
    """
    モデルの入力 (すべて log。列は PARAM_KEYS の順)
    """

    return np.log(np.asarray(P, dtype=np.float64)).astype(np.float32)


def stop_set():
    """
    停止判定用の固定点 (再開しても同じ点になるように、専用の乱数で作る)
    """

    return to_features(
        sample_space(np.random.default_rng([SEED, 999]), N_STOP)
    )


# ============================================================
# モデル
# ============================================================

def build_mlp(seed):
    """
    cascade/active_sim_cascade.py の MLP と同じ構成
    """

    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 64, 32),
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        ),
    )


def balance(X, y, seed):
    """
    ok の点を繰り返して ok / not ok の数を揃える。
    ok の点は全体の 1% 未満なので、そのまま学習すると「すべて 0」と予測してしまう
    """

    pos = np.flatnonzero(y == 1)
    n_rep = (len(y) - len(pos)) // max(len(pos), 1) - 1
    if len(pos) == 0 or n_rep <= 0:
        return X, y

    idx = np.r_[np.arange(len(y)), np.repeat(pos, n_rep)]
    np.random.default_rng(seed).shuffle(idx)
    return X[idx], y[idx]


def fit_one(X, y, seed):
    X, y = balance(X, y, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = build_mlp(seed)
        model.fit(X, y)
    return model


def fit_ensemble(X, y, seed, n_models):
    return Parallel(n_jobs=n_models)(
        delayed(fit_one)(X, y, seed * 100 + k)
        for k in range(n_models)
    )


def predict_ensemble(models, X):
    return np.concatenate([
        np.mean(
            [m.predict_proba(X[i:i + PREDICT_CHUNK])[:, 1] for m in models],
            axis=0
        )
        for i in range(0, len(X), PREDICT_CHUNK)
    ])


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ============================================================
# 次に回す点の選び方
# ============================================================

def select_active(rng, models, n):
    """
    境界付近 (k-means で散らす) の点と、ランダムな点を分けて返す
    """

    n_rand = int(round(n * RANDOM_FRAC))
    n_act = n - n_rand

    P_cand = sample_space(rng, N_CANDIDATES)
    score = entropy(predict_ensemble(models, to_features(P_cand)))

    n_top = min(len(P_cand), n_act * TOP_FACTOR)
    top = np.argpartition(-score, n_top - 1)[:n_top]

    X_top = StandardScaler().fit_transform(to_features(P_cand[top]))
    km = MiniBatchKMeans(
        n_clusters=n_act,
        init="random",
        batch_size=min(n_top, max(4096, 3 * n_act)),
        n_init=1,
        random_state=int(rng.integers(1 << 31)),
    ).fit(X_top)

    order = np.lexsort((-score[top], km.labels_))
    first = np.r_[True, km.labels_[order][1:] != km.labels_[order][:-1]]
    picked = top[order[first]]

    if len(picked) < n_act:
        rest = np.setdiff1d(top[np.argsort(-score[top])], picked,
                            assume_unique=True)
        picked = np.r_[picked, rest[:n_act - len(picked)]]

    return P_cand[picked], sample_space(rng, n_rand)


# ============================================================
# シミュレーション
# ============================================================

def repair_tail(path):
    """
    書きかけで止まった最後の行 (改行で終わっていない行) を消す
    """

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    with open(path, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        f.truncate(data.rfind(b"\n") + 1)


def init_csv(path, fieldnames):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def simulate_batch(executor, template, work_dir, P, roles, preds, rnd):
    """
    P の各点を並列にシミュレーションし、終わった順に samples.csv に追記する。
    戻り値: 成功した点の数
    """

    futures = {}
    for i, p in enumerate(P):
        v = dict(zip(PARAM_KEYS, (float(x) for x in p)))
        fut = executor.submit(bgr.simulate_point, template, work_dir, v)
        futures[fut] = (i, v)

    n_ok = 0
    t0 = time.perf_counter()

    with open(SAMPLES_PATH, "a", newline="") as f, \
            open(FAILED_PATH, "a", newline="") as ff:

        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS,
                                extrasaction="ignore")
        failed = csv.DictWriter(
            ff, fieldnames=PARAM_KEYS + ["round", "role", "error"]
        )

        for k, fut in enumerate(as_completed(futures), 1):
            i, v = futures[fut]

            try:
                row, log = fut.result()
            except Exception as e:
                row, log = None, f"ERROR: {e}"

            if row is None:
                err = log.strip().splitlines()[-1] if log.strip() else "unknown"
                failed.writerow({**v, "round": rnd, "role": roles[i],
                                 "error": err})
                print(f"FAILED {v}: {err}")
            else:
                row.update(round=rnd, role=roles[i], pred_before=preds[i])
                writer.writerow(row)
                n_ok += 1

            if k % 500 == 0 or k == len(futures):
                f.flush()
                ff.flush()
                el = time.perf_counter() - t0
                print(
                    f"    sim {k}/{len(futures)}  ({el:.0f}s, "
                    f"残り約 {el / k * (len(futures) - k):.0f}s)",
                    flush=True
                )

        os.fsync(f.fileno())

    return n_ok


def unique_points(P, roles, preds):
    """
    丸めで同じ値になった点を1つにする
    """

    _, idx = np.unique(P, axis=0, return_index=True)
    idx = np.sort(idx)
    return P[idx], [roles[i] for i in idx], np.asarray(preds)[idx]


# ============================================================
# 記録
# ============================================================

def load_samples():
    if not os.path.exists(SAMPLES_PATH):
        return None

    df = pd.read_csv(
        SAMPLES_PATH,
        usecols=PARAM_KEYS + [LABEL, "round", "role", "pred_before"],
    )
    df = df.dropna(subset=PARAM_KEYS + [LABEL])

    return df if len(df) else None


def load_rounds():
    if not os.path.exists(ROUNDS_PATH):
        return []
    return pd.read_csv(ROUNDS_PATH).to_dict("records")


def stop_pred_path(rnd):
    return os.path.join(STOP_PRED_DIR, f"round_{rnd}.npy")


def change_rate(pred, prev):
    """
    前ラウンドと今ラウンドのどちらかで ok と予測された点のうち、予測が変わった割合
    (どちらのラウンドでも ok と予測された点が無いときは nan にして、停止に数えない)
    """

    union = (pred | prev).sum()
    return float((pred != prev).sum() / union) if union else np.nan


def estimate_error(samples, rnd):
    s = samples[
        (samples["round"] > rnd - STOP_WINDOW)
        & (samples["role"] == "random")
        & samples["pred_before"].notna()
    ]
    if len(s) == 0:
        return np.nan, 0

    wrong = (s["pred_before"] >= 0.5).astype(int) != s[LABEL].astype(int)
    return float(wrong.mean()), len(s)


def load_test():
    """
    bgr.py の全探索の結果 (評価専用)。PARAM_SPACE の範囲内の行を TEST_SAMPLES 行に間引く
    """

    if not USE_TEST or not os.path.exists(TEST_CSV):
        return None

    df = pd.read_csv(TEST_CSV, usecols=PARAM_KEYS + [LABEL])

    inside = np.ones(len(df), dtype=bool)
    for k, (lo, hi, _, _) in PARAM_SPACE.items():
        inside &= (df[k] >= lo - 1e-6) & (df[k] <= hi + 1e-6)
    df = df[inside]

    if len(df) > TEST_SAMPLES:
        df = df.sample(n=TEST_SAMPLES, random_state=TEST_SEED)

    print(f"評価用の全探索データ: {len(df)} 行 (ok 率 {df[LABEL].mean():.4f})")

    return to_features(df[PARAM_KEYS].to_numpy()), df[LABEL].to_numpy()


def test_scores(pred, y):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan

    return {
        "test_acc": float((pred == (y == 1)).mean()),
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
    }


def plot_rounds(rounds, path):
    """
    点数 vs change_rate / est_err / 全探索の結果での誤り率と 1 − F1
    (コンテナに日本語フォントが無いので、図の文字は英語にする)
    """

    df = pd.DataFrame(rounds)

    # ok の点が見つからずモデルを一度も学習していないと、log 軸に描くものが無い
    if not (df[["change_rate", "est_err"]] > 0).any().any():
        print("rounds.png: 描く値が無いので省略します")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(df["n_labeled"], df["change_rate"], marker="o", ms=3,
            color="#d62728", label="change_rate (stop criterion)")
    ax.plot(df["n_labeled"], df["est_err"], marker="s", ms=3,
            color="#1f77b4", label="est_err (error on random points)")

    if "test_acc" in df and df["test_acc"].notna().any():
        ax.plot(df["n_labeled"], 1 - df["test_acc"], ls="--",
                color="k", label="error on test samples")
        ax.plot(df["n_labeled"], 1 - df["test_f1"], ls="-.",
                color="#2ca02c", label="1 - F1 on test samples")

    ax.axhline(TARGET_ERR, color="gray", ls=":",
               label=f"threshold {TARGET_ERR}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of simulations")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STOP_PRED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)
    if not spice_template or not os.path.exists(spice_template):
        print("Failed to generate base netlist. Please check xschem execution.")
        return

    with open(spice_template, "r") as src:
        template = src.read()

    num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))

    test = load_test()
    X_stop = stop_set()

    for path in [SAMPLES_PATH, FAILED_PATH]:
        repair_tail(path)
    init_csv(SAMPLES_PATH, SAMPLE_FIELDS)
    init_csv(FAILED_PATH, PARAM_KEYS + ["round", "role", "error"])

    samples = load_samples()
    rounds = load_rounds()

    print(
        f"方法 {STRATEGY}, 種 {SEED}, 出力先 {OUT_DIR}\n"
        f"停止条件: "
        + (f"change_rate < {TARGET_ERR} が {STOP_PATIENCE} ラウンド連続, "
           if STRATEGY == "active" else "")
        + (f"全探索の結果での F1 >= {STOP_F1}, " if STOP_F1 else "")
        + f"上限 {BUDGET} 回, 並列数 {num_workers}"
    )
    if samples is not None:
        print(f"再開: シミュレーション済み {len(samples)} 点, "
              f"ラウンド {int(samples['round'].max())} から")

    work_dir = tempfile.mkdtemp(prefix="bgr_al_",
                                dir=os.environ.get("WORK_DIR"))

    start = time.perf_counter()
    sim_s = np.nan
    models = None

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:

            if samples is None:
                print(f"\n初期サンプル {N_INIT} 点をシミュレーション中 ...")
                t0 = time.perf_counter()
                P = sample_space(np.random.default_rng([SEED, 0]), N_INIT)
                P, roles, preds = unique_points(
                    P, ["random"] * len(P), np.full(len(P), np.nan)
                )
                simulate_batch(executor, template, work_dir, P, roles,
                               preds, 0)
                sim_s = time.perf_counter() - t0
                samples = load_samples()
                if samples is None:
                    print("初期サンプルのシミュレーションがすべて失敗しました")
                    return

            while True:
                rnd = int(samples["round"].max())

                rounds = [r for r in rounds if r["round"] < rnd]

                t0 = time.perf_counter()

                X = to_features(samples[PARAM_KEYS].to_numpy())
                y = samples[LABEL].to_numpy().astype(int)

                # 片方のクラスしか無いと学習できないので、ランダムに足して待つ
                if y.min() == y.max():
                    models = None
                else:
                    models = fit_ensemble(X, y, SEED * 1000 + rnd,
                                          ENSEMBLE_SIZE)
                t_fit = time.perf_counter() - t0

                est_err, n_est = estimate_error(samples, rnd)
                n_failed = len(pd.read_csv(FAILED_PATH))

                row = {
                    "round": rnd,
                    "n_labeled": len(samples),
                    "n_failed": n_failed,
                    "ok_ratio_labeled": y.mean(),
                    "change_rate": np.nan,
                    "est_err": est_err,
                    "n_est": n_est,
                    "test_acc": np.nan,
                    "test_f1": np.nan,
                    "test_precision": np.nan,
                    "test_recall": np.nan,
                    "fit_s": t_fit,
                    "sim_s": sim_s,
                    "elapsed_s": time.perf_counter() - start,
                }

                if models is not None:
                    pred = predict_ensemble(models, X_stop) >= 0.5
                    np.save(stop_pred_path(rnd), pred)

                    if os.path.exists(stop_pred_path(rnd - 1)):
                        prev = np.load(stop_pred_path(rnd - 1))
                        row["change_rate"] = change_rate(pred, prev)

                    if test is not None:
                        p = predict_ensemble(models, test[0]) >= 0.5
                        row.update(test_scores(p, test[1]))

                rounds.append(row)
                pd.DataFrame(rounds).to_csv(ROUNDS_PATH, index=False)

                print(
                    f"[round {rnd}] n={row['n_labeled']}  "
                    f"change={row['change_rate']:.4f}  "
                    f"est_err={est_err:.4f} ({n_est}点)  "
                    f"test_acc={row['test_acc']:.4f}  "
                    f"test_f1={row['test_f1']:.4f}  "
                    f"ok率={row['ok_ratio_labeled']:.3f}  "
                    f"失敗 {n_failed}  "
                    f"(学習 {t_fit:.0f}s / シミュ {sim_s:.0f}s)",
                    flush=True
                )

                recent = [r["change_rate"] for r in rounds[-STOP_PATIENCE:]]
                if (STRATEGY == "active"
                        and len(recent) == STOP_PATIENCE
                        and all(c < TARGET_ERR for c in recent)):
                    print(f"\n停止: change_rate < {TARGET_ERR} が "
                          f"{STOP_PATIENCE} ラウンド続いた")
                    break

                if STOP_F1 is not None and row["test_f1"] >= STOP_F1:
                    print(f"\n停止: 全探索の結果での F1 が {STOP_F1} に達した")
                    break

                if len(samples) >= BUDGET:
                    print(f"\n停止: シミュレーション回数が上限 {BUDGET} に達した")
                    break

                n_add = min(
                    max(BATCH, int(len(samples) * BATCH_GROWTH)),
                    BUDGET - len(samples)
                )
                rng = np.random.default_rng([SEED, rnd + 1])

                if models is None or STRATEGY == "random":
                    P = sample_space(rng, n_add)
                    roles = ["random"] * len(P)
                    preds = (np.full(len(P), np.nan) if models is None
                             else predict_ensemble(models, to_features(P)))
                else:
                    P_act, P_rand = select_active(rng, models, n_add)
                    P = np.r_[P_act, P_rand]
                    roles = ["active"] * len(P_act) + ["random"] * len(P_rand)
                    preds = predict_ensemble(models, to_features(P))

                P, roles, preds = unique_points(P, roles, preds)

                print(f"  次の {len(P)} 点をシミュレーション中 ...")
                t0 = time.perf_counter()
                n_ok = simulate_batch(executor, template, work_dir, P,
                                      roles, preds, rnd + 1)
                sim_s = time.perf_counter() - t0

                if n_ok == 0:
                    print("このラウンドのシミュレーションがすべて失敗したので止めます")
                    break

                samples = load_samples()

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if models is not None:
        dump(models, os.path.join(MODEL_DIR, "ensemble.joblib"))

    if rounds:
        plot_rounds(rounds, os.path.join(OUT_DIR, "rounds.png"))

    elapsed = time.perf_counter() - start
    print(f"\nシミュレーション回数: {len(samples)} (失敗 {rounds[-1]['n_failed']})")
    print(f"経過時間 (今回の実行): {elapsed / 3600:.2f} h")
    print(f"保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
