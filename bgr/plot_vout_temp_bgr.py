import os
import re
import uuid
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from bgr import (
    export_netlist,
    modify_netlist_content,
    run_ngspice,
    PARAM_KEYS,
    LABEL,
    VOUT_TOL,
    SCH_PATH,
    NETLIST_DIR,
    STUDY_DIR,
    RESULTS_CSV,
)


# ============================================================
# bgr.py の全探索の結果 (bgr_results.csv) から、動作する (ok = 1) 設計のうち
# vout の変動 (spread) が小さいものを N_PICK 個選び、もう一度シミュレーションして
# 温度 vs vout を描く
#
#   bgr_results.csv には 0〜100 ℃ の vout の最大・最小・平均しか無いので、
#   選んだ設計だけ ngspice を回し直して vout(T) を wrdata で書き出す。
#   (ネットリストの .control を差し替える。回路図 BGR_al.sch は変えない)
#
#   出力 (study/bgr_vout_temp/):
#     picked.csv     : 選んだ設計 (bgr_results.csv の行)
#     curves.csv     : 温度ごとの vout (列: temp_C, #1, #2, ...)
#     vout_temp.png  : 左: vout / 右: 平均からのずれ [mV] (点線: ±VOUT_TOL)
#
#   コンテナ (ngspice がある側) で実行する想定 (図の文字は英語)。
#
#   環境変数:
#     N_PICK      : 選ぶ設計の数 (既定 10)
#     NUM_WORKERS : ngspice の並列数 (既定は CPU コア数)
#     WORK_DIR    : 作業用ネットリストを置く親フォルダ
# ============================================================


OUT_DIR = os.path.join(STUDY_DIR, "bgr_vout_temp")

N_PICK = int(os.environ.get("N_PICK", 10))

# 温度の掃引 [℃] (全探索は 2 ℃ 刻み。曲線をなめらかにするため 1 ℃ 刻みにする)
TEMP_START, TEMP_STOP, TEMP_STEP = 0, 100, 1


# ============================================================
# 設計の選び方
# ============================================================

def pick_designs(n):
    """
    ok = 1 の行を spread の小さい順に n 個
    """

    df = pd.read_csv(RESULTS_CSV)
    ok = df[df[LABEL] == 1].sort_values("spread_mV")

    print(f"全探索 {len(df)} 行のうち ok {len(ok)} 行から {n} 個を選ぶ")

    return ok.head(n).reset_index(drop=True)


# ============================================================
# シミュレーション
# ============================================================

def replace_control(content, data_path):
    """
    ネットリストの .control 〜 .endc を、vout(T) を data_path に書き出すものに差し替える
    """

    control = (
        ".control\n"
        "set num_threads=1\n"
        "save vout\n"
        f"dc temp {TEMP_START} {TEMP_STOP} {TEMP_STEP}\n"
        f"wrdata {data_path} v(vout)\n"
        ".endc"
    )

    content, count = re.subn(
        r'^\s*\.control\b.*?^\s*\.endc\b',
        lambda _: control,
        content,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if count == 0:
        raise RuntimeError(".control がネットリスト内に見つかりません")

    return content


def read_wrdata(path):
    """
    wrdata の出力 (1 列目: 温度, 2 列目: vout) を読む
    """

    rows = []
    with open(path) as f:
        for line in f:
            try:
                values = [float(v) for v in line.split()]
            except ValueError:
                continue    # 見出しの行など
            if len(values) >= 2:
                rows.append(values[:2])

    data = np.array(rows)
    return data[:, 0], data[:, 1]


def simulate_curve(template, work_dir, params):
    """
    1 つの設計の vout(T)。失敗したら None
    """

    name = os.path.join(work_dir, f"bgr_{uuid.uuid4().hex}")
    spice_path, data_path = name + ".spice", name + ".dat"

    try:
        content = modify_netlist_content(template, params)
        content = replace_control(content, data_path)

        with open(spice_path, "w") as f:
            f.write(content)

        success, stdout = run_ngspice(spice_path)
        if not success or not os.path.exists(data_path):
            print(f"FAILED {params}\n{stdout[-500:]}")
            return None

        return read_wrdata(data_path)

    finally:
        for path in (spice_path, data_path):
            if os.path.exists(path):
                os.remove(path)


# ============================================================
# 図
# ============================================================

def label_of(i, d):
    return (
        f"#{i + 1}: W={d['W_um']:g} L={d['L_um']:g} "
        f"R={d['R1_ohm']:g}/{d['R2_ohm']:g}/{d['R3_ohm']:g} "
        f"m={d['m1']:g}/{d['m2']:g}/{d['m3']:g}"
    )


def plot_curves(picked, curves, path):
    """
    左: vout / 右: 各設計の平均からのずれ [mV]
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, d in picked.iterrows():
        if i not in curves:
            continue
        t, v = curves[i]
        color = colors[i % 10]
        label = label_of(i, d)

        axes[0].plot(t, v, color=color, lw=1.2, label=label)
        axes[1].plot(t, (v - v.mean()) * 1e3, color=color, lw=1.2,
                     label=f"#{i + 1}: spread {(v.max() - v.min()) * 1e3:.3f} mV")

    axes[0].set_ylabel("vout [V]")
    axes[0].set_title("vout vs temperature")

    for s in (-1, 1):
        axes[1].axhline(s * VOUT_TOL * 1e3, color="gray", ls=":", lw=1)
    axes[1].set_ylabel("vout - mean [mV]")
    axes[1].set_title(f"deviation from mean (dotted: +/-{VOUT_TOL * 1e3:g} mV)")

    for ax in axes:
        ax.set_xlabel("temperature [C]")
        ax.grid(alpha=0.3)

    # 凡例 (パラメータが長いので図の下に置く)
    axes[0].legend(fontsize=7, loc="upper center",
                   bbox_to_anchor=(0.5, -0.12), ncol=1)
    axes[1].legend(fontsize=7, loc="upper center",
                   bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    picked = pick_designs(N_PICK)
    picked.to_csv(os.path.join(OUT_DIR, "picked.csv"), index=False)

    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)
    if not spice_template or not os.path.exists(spice_template):
        print("Failed to generate base netlist. Please check xschem execution.")
        return

    with open(spice_template, "r") as src:
        template = src.read()

    num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))
    work_dir = tempfile.mkdtemp(prefix="bgr_plot_",
                                dir=os.environ.get("WORK_DIR"))

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                i: executor.submit(
                    simulate_curve, template, work_dir,
                    {k: float(d[k]) for k in PARAM_KEYS}
                )
                for i, d in picked.iterrows()
            }
            curves = {i: f.result() for i, f in futures.items()}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    curves = {i: c for i, c in curves.items() if c is not None}
    if not curves:
        print("すべてのシミュレーションが失敗しました")
        return

    # 温度ごとの vout を 1 つの表にする
    t = next(iter(curves.values()))[0]
    table = pd.DataFrame({"temp_C": t})
    for i, (_, v) in curves.items():
        table[f"#{i + 1}"] = v
    table.to_csv(os.path.join(OUT_DIR, "curves.csv"), index=False)

    plot_curves(picked, curves, os.path.join(OUT_DIR, "vout_temp.png"))

    # 全探索 (2 ℃ 刻み) と今回 (1 ℃ 刻み) の spread を並べる
    for i, d in picked.iterrows():
        if i in curves:
            v = curves[i][1]
            print(f"#{i + 1}: spread 全探索 {d['spread_mV']:.3f} mV / "
                  f"今回 {(v.max() - v.min()) * 1e3:.3f} mV, "
                  f"vout 平均 {v.mean():.4f} V")

    print(f"保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
