import os
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import active_sim_opamp as base
from OPAMP import STUDY_DIR


# ============================================================
# active_sim_opamp.py (MLP での能動学習) の LightGBM 版
#
#   ngspice の回し方、設計空間 (PARAM_SPACE)、点の選び方、停止条件、再開の仕組みは
#   active_sim_opamp.py をそのまま使い、「モデル」と「保存先」だけを差し替える。
#   (active_sim_opamp_svm.py と同じ作り方)
#
#   モデル:
#     LightGBM を ENSEMBLE_SIZE 個 (MLP 版と同じ 5 個)。乱数と、木ごとに使う行・列の
#     間引き (bagging) を変えて、モデル同士の意見の割れ方から「境界付近」を見つける。
#     次に回す点の選び方は MLP 版と同じ (平均確率のエントロピーが大きい点)。
#
#   入力に「電流密度」を足す:
#     決定木は1つの入力ごとに区切るので、10個の入力 (W, L, Ibias, Vin) のままだと
#     「Ibias と W/L の組み合わせ」で決まる境界をうまく表せない。
#     そこで、各トランジスタに流れる電流 / (W/L) の log を入力に足す (add_current_density)。
#     MLP 版で集めた点で学習し、ランダムサンプリングの点 41763 点で評価した誤り率:
#                                入力 10 個     + 電流密度
#       ランダムな 30000 点で学習     0.044        0.011
#       能動学習の 41763 点で学習     0.067        0.0069
#     (足した列はモデルの中で計算するので、samples.csv などの形は MLP 版と同じ)
#
#   ハイパーパラメータ: num_leaves 31 / 63、n_estimators 1000 / 2000 で誤り率はほとんど
#   変わらなかったので、軽い設定にした。
#
#   最初のランダムな N_INIT 点は MLP 版と同じ乱数で作るので、同じ点から始まる。
#
#   出力 (study/opamp_active_sim_lgbm/): MLP 版と同じ構成
#     samples.csv, failed.csv, rounds.csv, rounds.png, stop_pred/
#     models/ensemble.joblib : 最後に学習した LightGBM (ENSEMBLE_SIZE 個のリスト)
#       (読み込むときは、このファイルと同じフォルダで import できる状態にすること。
#        電流密度を計算する関数 add_current_density をこのファイルから読むため)
#
#   環境変数 (MLP 版と共通のもの以外):
#     LGBM_JOBS : LightGBM の学習のスレッド数 (既定は CPU コア数)
# ============================================================


# ============================================================
# 設定
# ============================================================

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,          # 木ごとに使う行の割合 (モデルごとに違う行になる)
    subsample_freq=1,
    colsample_bytree=0.8,   # 木ごとに使う入力の割合
    verbose=-1,
)

LGBM_JOBS = int(os.environ.get("LGBM_JOBS", os.cpu_count() or 1))


# ============================================================
# 保存先 (MLP 版の結果を上書きしないよう、別のフォルダにする)
# ============================================================

OUT_DIR = os.path.join(
    STUDY_DIR,
    "opamp_active_sim_lgbm"
)

SAMPLES_PATH = os.path.join(OUT_DIR, "samples.csv")
FAILED_PATH = os.path.join(OUT_DIR, "failed.csv")
ROUNDS_PATH = os.path.join(OUT_DIR, "rounds.csv")
STOP_PRED_DIR = os.path.join(OUT_DIR, "stop_pred")
MODEL_DIR = os.path.join(OUT_DIR, "models")


# ============================================================
# 電流密度の入力
# ============================================================

def add_current_density(X):
    """
    to_features の出力 (W, L, Ibias は log、Vin はそのまま。列は PARAM_KEYS の順) に、
    各トランジスタの log(電流 / (W/L)) と、テール・2段目の log(電流) を足す。

    電流はミラーの比から求める (チャネル長変調などは無視した近似):
      M8, M9   : Ibias                      (M9 は M8 と同じ W)
      M5       : Ibias × Wtail / Wbp        (テール)
      M1～M4   : M5 の電流の半分
      M6, M7   : Ibias × W6 / Wbp           (2段目)
      M10, M11 : Ibias                      (M9 の電流)
    """

    X = np.asarray(X, dtype=np.float64)
    c = {k: X[:, j] for j, k in enumerate(base.PARAM_KEYS)}

    L = c["L_um"]
    ib = c["Ibias_uA"]
    i5 = ib + c["Wtail_um"] - c["Wbp_um"]
    i6 = ib + c["W6_um"] - c["Wbp_um"]

    extra = [
        i5 - np.log(2) - (c["Win_um"] - L),     # M1, M2
        i5 - np.log(2) - (c["Wload_um"] - L),   # M3, M4
        i5 - (c["Wtail_um"] - L),               # M5
        i6 - (c["W6_um"] - L),                  # M6
        i6 - (c["W7_um"] - L),                  # M7
        ib - (c["Wbp_um"] - L),                 # M8, M9
        ib - (c["Wbn_um"] - L),                 # M10, M11
        i5,
        i6,
    ]

    return np.column_stack([X] + extra).astype(np.float32)


# ============================================================
# モデル
# ============================================================

def build_lgbm(seed):
    return make_pipeline(
        FunctionTransformer(add_current_density),
        LGBMClassifier(random_state=seed, n_jobs=LGBM_JOBS, **LGBM_PARAMS),
    )


def fit_lgbm(X, y, seed, n_models):
    """
    MLP 版の fit_ensemble と同じ呼び方で、乱数だけ変えた LightGBM を n_models 個学習する。
    (1個ずつ全コアで学習する。1個あたり数秒)
    予測は MLP 版の predict_ensemble (各モデルの確率の平均) をそのまま使う
    """

    return [build_lgbm(seed * 100 + k).fit(X, y) for k in range(n_models)]


# ============================================================
# active_sim_opamp.py のモデルと保存先を差し替える
#   (active_sim_opamp.py の関数は、実行するときにこれらの名前を読みに行くので、
#    差し替えた後に base.main() を呼べば LightGBM 版として動く)
# ============================================================

def use_lgbm():

    base.OUT_DIR = OUT_DIR
    base.SAMPLES_PATH = SAMPLES_PATH
    base.FAILED_PATH = FAILED_PATH
    base.ROUNDS_PATH = ROUNDS_PATH
    base.STOP_PRED_DIR = STOP_PRED_DIR
    base.MODEL_DIR = MODEL_DIR

    base.fit_ensemble = fit_lgbm


if __name__ == "__main__":

    # このファイルを「active_sim_opamp_lgbm」として読み直してから使う。
    # (そのまま使うと、保存したモデルの add_current_density が "__main__" のものになり、
    #  plot や比較のスクリプトからモデルを読み込めなくなるため)
    import active_sim_opamp_lgbm as lgbm

    lgbm.use_lgbm()

    print(
        f"LightGBM 版 ({lgbm.LGBM_PARAMS['n_estimators']} 木, "
        f"num_leaves {lgbm.LGBM_PARAMS['num_leaves']}, {base.ENSEMBLE_SIZE} 個, "
        f"入力 + 電流密度)\n"
        f"保存先: {lgbm.OUT_DIR}"
    )

    base.main()
