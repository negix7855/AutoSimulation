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

from cascade import (
    export_netlist,
    simulate_point,
    DEVICE_PARAMS,
    SCH_PATH,
    NETLIST_DIR,
    STUDY_DIR,
)


# ============================================================
# カスコード増幅回路 (cascade.sch) の all_sat の境界を、
# ngspice を実際に回しながら能動学習で求める
#
#   opamp/active_sim_opamp.py と同じ方法をカスコード回路に当てる。
#   シミュレーションは cascade.py の simulate_point を使う (ネットリストの作り方・飽和判定は同じ)。
#
#   流れ (1ラウンド):
#     1. それまでにシミュレーションした点で MLP を ENSEMBLE_SIZE 個学習する
#     2. 停止判定用に固定した点 N_STOP 点で予測し、前ラウンドと予測が変わった点の割合
#        (change_rate) を求める。TARGET_ERR を STOP_PATIENCE ラウンド続けて下回ったら終了
#     3. 次に回す点を選ぶ
#          - 設計空間から新しく N_CANDIDATES 点を作り、平均確率のエントロピーが大きい
#            (= 境界付近で、モデル同士の意見も割れている) 点を k-means で散らして選ぶ
#          - RANDOM_FRAC の割合は設計空間からランダムに選ぶ (見落とし対策と est_err 用)
#     4. 選んだ点を ngspice で並列にシミュレーションし、結果を samples.csv に追記する
#
#   入力: W1, W2, L1, L2 (M1, M2 別々), Vb, Vin の6次元。RD は cascade.py のグリッドと同じ値に固定。
#   範囲 (PARAM_SPACE) は、opamp での結果に合わせて既存グリッドの端から余裕をとる。
#     W : グリッド 5.6–44.8 um → 2.5–100 um (log で一様)
#     L : グリッド 0.28–1.12 um → 0.28–2.5 um (log で一様。0.28 未満はモデルの範囲外)
#     Vb: グリッド 0.5–2.98 V → 0–3.3 V (VDD まで)
#     Vin: グリッド 0.3–1.68 V → 0.05–2.5 V (cascade.py が Av = Vout / Vin を計算するので 0 は避ける)
#   広げられない L1, L2 の下限 (0.28 um) には、FACE_POINTS の割合だけ意図的に点を置く。
#
#   補助的な指標 (停止には使わない):
#     est_err  : 直近 STOP_WINDOW ラウンドで「ランダムに選んだ点」について、
#                シミュレーション前の予測 (その点を選んだ時点のアンサンブル) が外れていた割合
#     test_acc : 既存の全グリッドデータ cascade_results.csv での正解率 (TEST_SAMPLES 行に間引く)
#                (学習には使わない。AL_TEST=0 で評価しない)
#
#   途中で止めても、もう一度実行すれば続きから再開する。
#   (samples.csv と rounds.csv、stop_pred/ を読み直す。範囲や設定を変えたら OUT_DIR を変える)
#
#   出力 (study/cascade_active_sim/。random なら _random、AL_SEED が 0 以外なら _seed<N> を付ける):
#     samples.csv          : シミュレーションした点 (cascade.py と同じ列 + round, role, pred_before)
#     failed.csv           : シミュレーションに失敗した点
#     rounds.csv           : ラウンドごとの点数、change_rate、est_err、test_acc、時間
#     rounds.png           : 点数 vs change_rate / est_err / テスト誤り率
#     stop_pred/           : 停止判定用の点での各ラウンドの予測 (change_rate の計算と再開用)
#     models/ensemble.joblib : 最後に学習した MLP (ENSEMBLE_SIZE 個のリスト)
#
#   環境変数:
#     AL_TARGET   : 停止する change_rate のしきい値 (既定 0.003)
#     AL_BUDGET   : シミュレーション回数の上限
#     AL_TEST     : 0 にすると既存グリッドでの評価をしない
#     AL_SEED     : 乱数の種 (既定 0)。初期サンプル・候補点・停止判定用の点・MLP の初期値が変わる
#     AL_STRATEGY : active (既定) / random。random は比較用で、追加点をすべて一様ランダムに選ぶ
#                   (設計空間・MLP・評価は active と同じ)。random では change_rate で止めず、
#                   AL_STOP_ACC か AL_BUDGET で止める
#     AL_STOP_ACC : 既存グリッドでの accuracy がこの値以上になったら止める (比較実験用。
#                   random の既定 0.9957 = active (seed 0) の停止時。active では既定なし)
#     AL_GROWTH   : 1ラウンドの追加点数の増え方 (既定 0.1。random の既定 0.2)
#     NUM_WORKERS : ngspice の並列数 (cascade.py と同じ。既定は CPU コア数)
#     WORK_DIR    : 作業用ネットリストを置く親フォルダ (cascade.py と同じ)
# ============================================================


# ============================================================
# パス
# ============================================================

# 乱数の種 (0 のときは今までと同じ結果・同じフォルダになる)
SEED = int(os.environ.get("AL_SEED", 0))

# 点の選び方 (active: 能動学習, random: 比較用の一様ランダム)
STRATEGY = os.environ.get("AL_STRATEGY", "active")
if STRATEGY not in ("active", "random"):
    raise ValueError(f"AL_STRATEGY は active か random: {STRATEGY}")

OUT_DIR = os.path.join(
    STUDY_DIR,
    "cascade_active_sim"
    + ("_random" if STRATEGY == "random" else "")
    + (f"_seed{SEED}" if SEED else "")
)

SAMPLES_PATH = os.path.join(OUT_DIR, "samples.csv")
FAILED_PATH = os.path.join(OUT_DIR, "failed.csv")
ROUNDS_PATH = os.path.join(OUT_DIR, "rounds.csv")
STOP_PRED_DIR = os.path.join(OUT_DIR, "stop_pred")
MODEL_DIR = os.path.join(OUT_DIR, "models")

# 評価用の既存グリッドデータ
TEST_CSV = os.path.join(STUDY_DIR, "cascade_results.csv")


# ============================================================
# 設定
# ============================================================

# 負荷抵抗 [Ω] (cascade.py のグリッドと同じ値に固定)
RD_OHM = 10000

# 振る範囲: (下限, 上限, log で振るか, 丸める小数点以下の桁数)
PARAM_SPACE = {
    "W1_um": (2.5, 100.0, True, 2),    # M1 (入力)
    "W2_um": (2.5, 100.0, True, 2),    # M2 (カスコード)
    "L1_um": (0.28, 2.5, True, 2),     # M1 (0.28 未満はシミュレーションが失敗する)
    "L2_um": (0.28, 2.5, True, 2),     # M2
    "Vb_V": (0.0, 3.3, False, 3),      # M2 のゲート電圧
    "Vin_V": (0.001, 2.5, False, 3),    # M1 のゲート電圧
}

# 入力の列
PARAM_KEYS = list(PARAM_SPACE)

# 範囲を広げられない端 (L の下限はトランジスタモデルの限界) に、意図的に点を置く
#   {列: (値, 割合)} : sample_space で作る点のうち「割合」の点だけ、その列を「値」に固定する
#   (能動学習の候補点・ランダムな点・停止判定用の点のすべてに同じ割合で入る)
#   opamp では一様に振るだけだと L = 0.28 の点が約 1% しかなく、そこでの誤りが多く残ったため
FACE_POINTS = {
    "L1_um": (0.28, 0.10),
    "L2_um": (0.28, 0.10),
}

LABEL = "all_sat"

# samples.csv の列 (cascade.py の CSV と同じ並び + round, role, pred_before)
FIELDNAMES = (
    ["RD_ohm"] + PARAM_KEYS + ["Vout_V", "Av"]
    + [f"M{i}_{p}" for i in (1, 2) for p in DEVICE_PARAMS + ["sat"]]
    + [LABEL]
)
SAMPLE_FIELDS = FIELDNAMES + ["round", "role", "pred_before"]

# 最初にランダムに回す点数
N_INIT = 1000

# 1ラウンドで追加する点数 = max(BATCH, それまでの点数 × BATCH_GROWTH)
BATCH = 1000
BATCH_GROWTH = float(os.environ.get(
    "AL_GROWTH", 0.2 if STRATEGY == "random" else 0.1
))

# シミュレーション回数の上限 (change_rate で止まらなかった場合)
# random は active と同じ精度まで多くの点が要るので、上限を大きくしておく
BUDGET = int(os.environ.get(
    "AL_BUDGET", 1_000_000 if STRATEGY == "random" else 60000
))

# 既存グリッドでの accuracy がこの値以上になったら止める (比較実験用。None なら使わない)
_stop_acc = os.environ.get("AL_STOP_ACC", "0.9957" if STRATEGY == "random" else "")
STOP_ACC = float(_stop_acc) if _stop_acc else None

# 停止条件: change_rate が TARGET_ERR を STOP_PATIENCE ラウンド続けて下回ったら止める
TARGET_ERR = float(os.environ.get("AL_TARGET", 0.0030))
STOP_PATIENCE = 3

# 停止判定用に固定する点の数
N_STOP = 100_000

# est_err を何ラウンド分合算するか
STOP_WINDOW = 3

# MLP の個数
ENSEMBLE_SIZE = 5

# 1ラウンドの追加点のうち、ランダムに選ぶ割合
RANDOM_FRAC = 0.2

# 1ラウンドで新しく作ってスコアする候補点の数
N_CANDIDATES = 500_000

# k-means にかける上位候補の数 = 追加点数 × この倍率
TOP_FACTOR = 10

# 予測を何行ずつに分けるか
PREDICT_CHUNK = 500_000

# 既存グリッドでの評価
USE_TEST = os.environ.get("AL_TEST", "1") != "0"
TEST_SAMPLES = 1_000_000

# 評価用グリッドの間引き方は種によらず固定する (種どうしで同じ行で比べるため)
TEST_SEED = 0


# ============================================================
# 設計空間
# ============================================================

def sample_space(rng, n):
    """
    PARAM_SPACE の範囲から n 点を一様に作る (W, L は log で一様)。
    FACE_POINTS の列は、指定した割合の点だけ端の値に固定する。
    戻り値は (n, len(PARAM_KEYS)) の配列 (列は PARAM_KEYS の順)
    """

    u = rng.random((n, len(PARAM_KEYS)))
    P = np.empty_like(u)

    for j, k in enumerate(PARAM_KEYS):
        lo, hi, log, digits = PARAM_SPACE[k]
        if log:
            P[:, j] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
        else:
            P[:, j] = lo + u[:, j] * (hi - lo)
        P[:, j] = np.round(P[:, j], digits)

    for k, (value, frac) in FACE_POINTS.items():
        P[rng.random(n) < frac, PARAM_KEYS.index(k)] = value

    return P


def to_features(P):
    """
    W, L を log に変換した入力 (列は PARAM_KEYS の順)
    """

    X = np.asarray(P, dtype=np.float64).copy()

    for j, k in enumerate(PARAM_KEYS):
        if PARAM_SPACE[k][2]:
            X[:, j] = np.log(X[:, j])

    return X.astype(np.float32)


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
    opamp/active_sim_opamp.py の MLP と同じ構成
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


def fit_one(X, y, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = build_mlp(seed)
        model.fit(X, y)
    return model


def fit_ensemble(X, y, seed, n_models):
    """
    初期値だけ変えた MLP を n_models 個、並列に学習する
    """

    return Parallel(n_jobs=n_models)(
        delayed(fit_one)(X, y, seed * 100 + k)
        for k in range(n_models)
    )


def predict_ensemble(models, X):
    """
    各モデルの確率の平均を PREDICT_CHUNK 行ずつ計算してつなげる
    """

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

    # 候補点を設計空間から新しく作ってスコアする
    P_cand = sample_space(rng, N_CANDIDATES)
    score = entropy(predict_ensemble(models, to_features(P_cand)))

    # 境界寄りの上位を k-means で n_act 個に分け、各クラスタで最もスコアが高い点を取る
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

    # クラスタが空になって足りない分は、スコア順で埋める
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

    cascade.py の simulate_point は作業用ネットリストの名前を条件の値から作るので、
    同じ条件の点が同時に走るとファイルがぶつかる。呼び出す側で重複を除いておく。
    """

    futures = {}
    for i, p in enumerate(P):
        v = dict(zip(PARAM_KEYS, (float(x) for x in p)))
        fut = executor.submit(
            simulate_point, template, work_dir, RD_OHM,
            v["W1_um"], v["W2_um"], v["L1_um"], v["L2_um"],
            v["Vb_V"], v["Vin_V"]
        )
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
                # cascade.py のログは複数行なので、最後の行 (失敗の理由) だけ残す
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
    丸めで同じ値になった点を1つにする (作業用ファイルの衝突を避けるため)
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

    # 学習と est_err に使う列だけ読む (random では行数が多くなるため)
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


def estimate_error(samples, rnd):
    """
    直近 STOP_WINDOW ラウンドに「ランダムに選んだ点」で、
    シミュレーション前の予測が外れていた割合
    """

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
    既存の全グリッドデータ (評価専用)。RD = RD_OHM かつ PARAM_SPACE の範囲内の行を
    TEST_SAMPLES 行に間引く
    """

    if not USE_TEST or not os.path.exists(TEST_CSV):
        return None

    df = pd.read_csv(
        TEST_CSV,
        usecols=["RD_ohm"] + PARAM_KEYS + [LABEL],
        dtype={**{k: np.float32 for k in PARAM_KEYS}, LABEL: np.int8},
    )

    inside = np.isclose(df["RD_ohm"], RD_OHM)
    for k, (lo, hi, _, _) in PARAM_SPACE.items():
        inside &= (df[k] >= lo - 1e-6) & (df[k] <= hi + 1e-6)
    df = df[inside]

    if len(df) > TEST_SAMPLES:
        df = df.sample(n=TEST_SAMPLES, random_state=TEST_SEED)

    print(f"評価用グリッド: {len(df)} 行 (飽和率 {df[LABEL].mean():.3f})")

    return to_features(df[PARAM_KEYS].to_numpy()), df[LABEL].to_numpy()


def plot_rounds(rounds, path):
    """
    点数 vs change_rate / est_err / テスト誤り率
    (コンテナに日本語フォントが無いので、図の文字は英語にする)
    """

    df = pd.DataFrame(rounds)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(df["n_labeled"], df["change_rate"], marker="o", ms=3,
            color="#d62728", label="change_rate (stop criterion)")
    ax.plot(df["n_labeled"], df["est_err"], marker="s", ms=3,
            color="#1f77b4", label="est_err (error on random points)")

    if "test_acc" in df and df["test_acc"].notna().any():
        ax.plot(df["n_labeled"], 1 - df["test_acc"], ls="--",
                color="k", label="error on existing grid")

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

    # 基準となるネットリスト
    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)
    if not spice_template or not os.path.exists(spice_template):
        print("Failed to generate base netlist. Please check xschem execution.")
        return

    with open(spice_template, "r") as src:
        template = src.read()

    num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))

    test = load_test()
    X_stop = stop_set()

    # 再開の準備
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
        + (f"グリッドでの accuracy >= {STOP_ACC}, " if STOP_ACC else "")
        + f"上限 {BUDGET} 回, 並列数 {num_workers}"
    )
    if samples is not None:
        print(f"再開: シミュレーション済み {len(samples)} 点, "
              f"ラウンド {int(samples['round'].max())} から")

    # 作業用ネットリストの置き場 (cascade.py と同じ)
    work_dir = tempfile.mkdtemp(prefix="cascade_al_",
                                dir=os.environ.get("WORK_DIR"))

    start = time.perf_counter()
    sim_s = np.nan
    models = None

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:

            # 最初はランダムに N_INIT 点
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

                # 再開時、このラウンドの記録が既にあれば作り直す
                rounds = [r for r in rounds if r["round"] < rnd]

                t0 = time.perf_counter()

                X = to_features(samples[PARAM_KEYS].to_numpy())
                y = samples[LABEL].to_numpy().astype(int)

                # 片方のクラスしか無いと学習できないので、ランダムに足して待つ
                if y.min() == y.max():
                    models = None
                else:
                    # 種とラウンドの組で初期値を決める (SEED + rnd だと種どうしで重なる)
                    models = fit_ensemble(X, y, SEED * 1000 + rnd,
                                          ENSEMBLE_SIZE)
                t_fit = time.perf_counter() - t0

                est_err, n_est = estimate_error(samples, rnd)
                n_failed = len(pd.read_csv(FAILED_PATH))

                row = {
                    "round": rnd,
                    "n_labeled": len(samples),
                    "n_failed": n_failed,
                    "sat_ratio_labeled": y.mean(),
                    "change_rate": np.nan,
                    "est_err": est_err,
                    "n_est": n_est,
                    "test_acc": np.nan,
                    "fit_s": t_fit,
                    "sim_s": sim_s,
                    "elapsed_s": time.perf_counter() - start,
                }

                if models is not None:
                    pred = predict_ensemble(models, X_stop) >= 0.5
                    np.save(stop_pred_path(rnd), pred)

                    if os.path.exists(stop_pred_path(rnd - 1)):
                        prev = np.load(stop_pred_path(rnd - 1))
                        row["change_rate"] = float((pred != prev).mean())

                    if test is not None:
                        p = predict_ensemble(models, test[0]) >= 0.5
                        row["test_acc"] = float((p == test[1]).mean())

                rounds.append(row)
                pd.DataFrame(rounds).to_csv(ROUNDS_PATH, index=False)

                print(
                    f"[round {rnd}] n={row['n_labeled']}  "
                    f"change={row['change_rate']:.4f}  "
                    f"est_err={est_err:.4f} ({n_est}点)  "
                    f"test_acc={row['test_acc']:.4f}  "
                    f"sat率={row['sat_ratio_labeled']:.3f}  "
                    f"失敗 {n_failed}  "
                    f"(学習 {t_fit:.0f}s / シミュ {sim_s:.0f}s)",
                    flush=True
                )

                # 停止判定: 直近 STOP_PATIENCE ラウンドの change_rate がすべて TARGET_ERR 未満
                # (random は比較用なので change_rate では止めない)
                recent = [r["change_rate"] for r in rounds[-STOP_PATIENCE:]]
                if (STRATEGY == "active"
                        and len(recent) == STOP_PATIENCE
                        and all(c < TARGET_ERR for c in recent)):
                    print(f"\n停止: change_rate < {TARGET_ERR} が "
                          f"{STOP_PATIENCE} ラウンド続いた")
                    break

                if STOP_ACC is not None and row["test_acc"] >= STOP_ACC:
                    print(f"\n停止: 既存グリッドでの accuracy が {STOP_ACC} に達した")
                    break

                if len(samples) >= BUDGET:
                    print(f"\n停止: シミュレーション回数が上限 {BUDGET} に達した")
                    break

                # 次に回す点
                n_add = min(
                    max(BATCH, int(len(samples) * BATCH_GROWTH)),
                    BUDGET - len(samples)
                )
                rng = np.random.default_rng([SEED, rnd + 1])

                if models is None or STRATEGY == "random":
                    P = sample_space(rng, n_add)
                    roles = ["random"] * len(P)
                    # シミュレーション前の予測 (est_err 用)
                    preds = (np.full(len(P), np.nan) if models is None
                             else predict_ensemble(models, to_features(P)))
                else:
                    P_act, P_rand = select_active(rng, models, n_add)
                    P = np.r_[P_act, P_rand]
                    roles = ["active"] * len(P_act) + ["random"] * len(P_rand)
                    # シミュレーション前の予測 (est_err 用)
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
        # 途中で止めた場合も一時フォルダを消す
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
