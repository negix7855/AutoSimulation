import os
import re
import csv
import itertools
import time
import uuid
import shutil
import tempfile
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# バンドギャップリファレンス (BGR_al.sch) のシミュレーションと、軽い全探索
# (cascade/cascade.py, opamp/OPAMP.py の BGR 版)
#
#   回路図 BGR_al.sch は BGR.sch に次を足したもの:
#     - 振る値を .param にした (W, L, R1, R2, R3, m1, m2, m3)
#     - スタートアップ回路 (MS1〜MS3)。電流ゼロの状態に落ちた点が
#       「vout が温度で変わらない = 動作」と誤判定されないようにするため
#     - バッチ実行用の control (0〜100 ℃ の dc temp で vout の最大・最小・平均と、
#       コアの電流の最小値を print する)
#
#   入力 (8 個):
#     W, L   : MOS 共通 (カレントミラー M3〜M5 と PTAT の M1, M2。m = mossize は固定)
#     R1〜R3 : 抵抗 [Ω]
#     m1〜m3 : バイポーラ Q1〜Q3 の並列数
#
#   動作の判定 (ok = 1):
#     - 0〜100 ℃ での vout の変動 (最大 − 最小) が 2 × VOUT_TOL 以内 (= 中心 ±2.5 mV)
#     - コアの電流の最小値が I_MIN 以上 (電流ゼロの状態に落ちていない)
#     - vout が VOUT_RANGE の中 (M5 が線形領域に入り vout ≈ VDD で平らになった点などを除く)
#
#   直接実行すると、GRID の全組み合わせをシミュレーションし、
#   study/bgr_results.csv に保存する (能動学習 active_sim_bgr.py の検証データ)。
#   結果は終わった順に追記するので、途中で止めても、もう一度実行すれば続きから再開する
#   (GRID を変えたら bgr_results.csv を消してから実行する)。
#
#   環境変数:
#     NUM_WORKERS : ngspice の並列数 (既定は CPU コア数)
#     WORK_DIR    : 作業用ネットリストを置く親フォルダ
# ============================================================


# ============================================================
# パス
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

STUDY_DIR = os.path.join(PROJECT_ROOT, "study")
NETLIST_DIR = os.path.join(STUDY_DIR, "netlists")

SCH_FILENAME = "BGR_al.sch"
SCH_PATH = os.path.join(STUDY_DIR, SCH_FILENAME)

RESULTS_CSV = os.path.join(STUDY_DIR, "bgr_results.csv")


# ============================================================
# 設定
# ============================================================

# 全探索で振る値 (この全組み合わせを回す)
GRID = {
    "W_um": [5.6, 11.2, 22.4, 44.8],
    "L_um": [0.28, 0.56, 1.12, 2.24],
    "R1_ohm": [1, 3, 10, 30, 100, 300, 1000],
    "R2_ohm": [50, 150, 500, 1500, 5000, 15000, 50000],
    "R3_ohm": [float(v) for v in np.round(np.geomspace(200, 200000, 21))],
    "m1": [1, 2, 4, 8],
    "m2": [2, 4, 8, 16, 32],
    "m3": [1, 2, 4, 8],
}

PARAM_KEYS = list(GRID)

# 列名 → 回路図の .param 名と単位の接尾辞
NETLIST_PARAMS = {
    "W_um": ("W", "u"),
    "L_um": ("L", "u"),
    "R1_ohm": ("R1", ""),
    "R2_ohm": ("R2", ""),
    "R3_ohm": ("R3", ""),
    "m1": ("m1", ""),
    "m2": ("m2", ""),
    "m3": ("m3", ""),
}

# 動作の判定
VOUT_TOL = 0.0025            # [V] 中心からの許容幅 (変動 = 最大 − 最小 ≤ 2 × VOUT_TOL)
I_MIN = 1e-7                 # [A] コアの電流の下限
VOUT_RANGE = (1.0, 1.4)      # [V] vout が入っているべき範囲

LABEL = "ok"

MEASURES = ["vmax", "vmin", "vavg", "iamin", "ibmin"]

FIELDNAMES = (
    PARAM_KEYS
    + ["vmax_V", "vmin_V", "vavg_V", "spread_mV", "iamin_A", "ibmin_A",
       "started", "in_range", LABEL]
)

env = os.environ.copy()


# ============================================================
# Xschem → SPICEネットリスト生成
# ============================================================

def export_netlist(sch_path, output_dir):
    """
    xschemをバッチモードで実行し、SPICEネットリストを生成する
    """

    os.makedirs(output_dir, exist_ok=True)

    cmd = ["xschem", "-q", "-n", "-s", "-o", output_dir, sch_path]

    print(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("xschem Error:")
        print(result.stderr)
        return None

    base_name = os.path.splitext(os.path.basename(sch_path))[0]

    return os.path.join(output_dir, f"{base_name}.spice")


# ============================================================
# .param の変更
# ============================================================

def modify_netlist_content(content, params):
    """
    params ({列名: 値}) の .param を書き換えて返す
    """

    for key, value in params.items():
        name, suffix = NETLIST_PARAMS[key]

        content, count = re.subn(
            rf'^(\s*\.param\s+{re.escape(name)}\s*=\s*)\S+',
            rf'\g<1>{value:g}{suffix}',
            content,
            flags=re.IGNORECASE | re.MULTILINE
        )

        if count == 0:
            print(f"WARNING: .param {name} がネットリスト内に見つかりません")

    return content


# ============================================================
# ngspice実行と結果の取り出し
# ============================================================

def run_ngspice(netlist_path):
    """
    ngspiceをバッチモードで実行する
    """

    result = subprocess.run(
        ["ngspice", "-b", netlist_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return False, result.stdout + result.stderr

    return True, result.stdout


def extract_measures(stdout_text):
    """
    control の print で出した

        vmax = 1.20140e+00

    のような行から値を取り出す (取れなかったものは入れない)
    """

    values = {}

    for name in MEASURES:
        match = re.search(
            rf'^\s*{name}\s*=\s*'
            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',
            stdout_text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        if match:
            values[name] = float(match.group(1))

    return values


def judge(m):
    """
    取り出した値から (started, in_range, ok) を決める
    """

    started = min(m["iamin"], m["ibmin"]) >= I_MIN
    in_range = VOUT_RANGE[0] <= m["vmin"] and m["vmax"] <= VOUT_RANGE[1]
    flat = (m["vmax"] - m["vmin"]) <= 2 * VOUT_TOL

    return int(started), int(in_range), int(started and in_range and flat)


# ============================================================
# 1条件分のシミュレーション (並列実行の単位)
# ============================================================

def simulate_point(template_content, work_dir, params):
    """
    1つの条件 (params: {列名: 値}) についてネットリスト作成 → ngspice実行 → 判定を行う。
    並列実行時に表示が混ざらないよう、ログは文字列で返す。

    戻り値: (row または None, ログ文字列)
    """

    log = [", ".join(f"{k}={v:g}" for k, v in params.items())]

    # 作業用ネットリスト (並列でも衝突しないように名前は毎回別にする)
    work_spice = os.path.join(work_dir, f"bgr_{uuid.uuid4().hex}.spice")

    try:
        content = modify_netlist_content(template_content, params)

        with open(work_spice, "w") as dst:
            dst.write(content)

        success, stdout = run_ngspice(work_spice)

        if not success:
            log.append("Simulation Failed")
            return None, "\n".join(log)

        m = extract_measures(stdout)

        missing = [name for name in MEASURES if name not in m]
        if missing:
            log.append(f"WARNING: {missing} が取得できませんでした")
            return None, "\n".join(log)

        started, in_range, ok = judge(m)

        row = {
            **params,
            "vmax_V": m["vmax"],
            "vmin_V": m["vmin"],
            "vavg_V": m["vavg"],
            "spread_mV": (m["vmax"] - m["vmin"]) * 1e3,
            "iamin_A": m["iamin"],
            "ibmin_A": m["ibmin"],
            "started": started,
            "in_range": in_range,
            LABEL: ok,
        }

        log.append(
            f"SUCCESS: vavg = {m['vavg']:.4f} V, "
            f"spread = {row['spread_mV']:.2f} mV, "
            f"I = {m['iamin']:.3g} A, ok = {ok}"
        )

    finally:
        if os.path.exists(work_spice):
            os.remove(work_spice)

    return row, "\n".join(log)


# ============================================================
# メイン (GRID の全探索)
# ============================================================

def main():

    num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))

    print(f"Working Directory: {BASE_DIR}")
    print(f"Target Schematic: {SCH_PATH}")

    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)
    if not spice_template or not os.path.exists(spice_template):
        print("Failed to generate base netlist. Please check xschem execution.")
        return

    print(f"Base netlist generated: {spice_template}")

    with open(spice_template, "r") as src:
        template = src.read()

    # 全条件の組み合わせ (順番は GRID で決まるので、番号 idx で再開できる)
    conditions = [
        dict(zip(PARAM_KEYS, (float(v) for v in values)))
        for values in itertools.product(*GRID.values())
    ]

    fields = ["idx"] + FIELDNAMES
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            done = {int(r["idx"]) for r in csv.DictReader(f)}
    else:
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    todo = [i for i in range(len(conditions)) if i not in done]

    # 作業用ネットリストはコンテナ内の一時フォルダ (/tmp/bgr_xxxx) に作る
    work_dir = tempfile.mkdtemp(prefix="bgr_", dir=os.environ.get("WORK_DIR"))

    print(
        f"\nTotal conditions: {len(conditions)} "
        f"(済み {len(done)}, 残り {len(todo)}), workers: {num_workers}\n"
        f"Work directory: {work_dir}"
    )

    start_time = time.perf_counter()
    n_ok = n_fail = 0

    try:
        # ngspiceは別プロセスで動くので、スレッドでも並列に実行される
        with ThreadPoolExecutor(max_workers=num_workers) as executor,                 open(RESULTS_CSV, "a", newline="") as f:

            writer = csv.DictWriter(f, fieldnames=fields)

            futures = {
                executor.submit(simulate_point, template, work_dir,
                                conditions[i]): i
                for i in todo
            }

            for k, fut in enumerate(as_completed(futures), 1):
                try:
                    row, log = fut.result()
                except Exception as e:
                    row, log = None, f"ERROR: {e}"

                if row is None:
                    n_fail += 1
                    print(f"FAILED idx={futures[fut]}: {log}")
                else:
                    writer.writerow({"idx": futures[fut], **row})
                    n_ok += row[LABEL]

                if k % 500 == 0 or k == len(futures):
                    f.flush()
                    el = time.perf_counter() - start_time
                    print(f"  {k}/{len(futures)}  ({el:.0f}s, "
                          f"残り約 {el / k * (len(futures) - k):.0f}s, "
                          f"ok {n_ok}, 失敗 {n_fail})", flush=True)

    finally:
        # 途中で止めた場合も一時フォルダを消す
        shutil.rmtree(work_dir, ignore_errors=True)

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n====================================")
    print(f"CSV saved: {RESULTS_CSV}")
    print(f"今回: 成功 {len(todo) - n_fail} / {len(todo)}, ok {n_ok}")
    print(f"Total simulation time: "
          f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("====================================")


if __name__ == "__main__":
    main()
