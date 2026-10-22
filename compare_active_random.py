import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 能動学習 (active) とランダムサンプリング (random) を、
# 「既存グリッドでの accuracy が目標に届くまでのシミュレーション回数と時間」で比べる
# (opamp/compare_random_opamp.py を回路によらず使えるようにしたもの)
#
#   使い方: python compare_active_random.py <回路>   (回路: opamp / 5tota / cascade / all)
#
#   読むもの (各回路の active_sim_*.py の出力 rounds.csv):
#     active : study/<基準名>/, study/<基準名>_seed<N>/
#     random : study/<基準名>_random/, study/<基準名>_random_seed<N>/
#
#   目標の accuracy: 99%, 99.5%, active の停止時の accuracy の平均 (小数点以下4桁で切り捨て)
#
#   時間の数え方 (実際に使うときにかかる時間で比べる):
#     active : 経過時間そのもの (毎ラウンドの学習も点選びに必要なので含める)
#     random : それまでのシミュレーション時間の合計 + 最後に1回だけ学習する時間
#
#   出力 (study/<回路>_active_vs_random/):
#     reach.csv   : 実行ごと・目標ごとの 届いた回数と時間 (届かなければ空欄)
#     curves.png  : 回数 vs 誤り率 / 時間 vs 誤り率
#
#   Windows 側で実行する想定 (図に日本語を使う)。
# ============================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "study"))

# 回路 → 出力フォルダの基準名
CIRCUITS = {
    "opamp": "opamp_active_sim_w2p5",
    "5tota": "5tota_active_sim",
    "cascade": "cascade_active_sim",
}

FIXED_TARGETS = [0.99, 0.995]


def find_runs(base_name):
    """
    [(方法, 種, rounds の DataFrame)]
    """

    runs = []
    pattern = re.compile(rf"{re.escape(base_name)}(_random)?(?:_seed(\d+))?")

    for d in sorted(glob.glob(os.path.join(STUDY_DIR, base_name + "*"))):
        m = pattern.fullmatch(os.path.basename(d))
        path = os.path.join(d, "rounds.csv")
        if m is None or not os.path.exists(path):
            continue

        strategy = "random" if m.group(1) else "active"
        seed = int(m.group(2) or 0)
        r = pd.read_csv(path)

        # 時間の列を足す (途中から再開した実行では、最初のラウンドの sim_s が空になる)
        sim = r["sim_s"].fillna(0).cumsum()
        r["time_s"] = (r["elapsed_s"] if strategy == "active"
                       else sim + r["fit_s"])
        r["test_err"] = 1 - r["test_acc"]

        runs.append((strategy, seed, r))

    return runs


def make_targets(runs):
    """
    99%, 99.5% と、active の停止時の accuracy の平均
    """

    finals = [r["test_acc"].iloc[-1] for s, _, r in runs if s == "active"]
    targets = list(FIXED_TARGETS)
    if finals:
        level = np.floor(np.mean(finals) * 1e4) / 1e4
        if level not in targets:
            targets.append(level)
    return sorted(targets)


def reach_table(runs, targets):
    """
    実行ごと・目標ごとに、初めて目標に届いたラウンドの回数と時間
    """

    rows = []
    for strategy, seed, r in runs:
        for t in targets:
            hit = r[r["test_acc"] >= t]
            row = {"strategy": strategy, "seed": seed, "target": t,
                   "n_sim": np.nan, "time_min": np.nan}
            if len(hit):
                row["n_sim"] = int(hit["n_labeled"].iloc[0])
                row["time_min"] = hit["time_s"].iloc[0] / 60
            rows.append(row)

        last = r.iloc[-1]
        print(f"  {strategy} seed {seed}: 最後 n={int(last['n_labeled'])}, "
              f"accuracy={last['test_acc']:.4f}, "
              f"時間 {last['time_s'] / 60:.1f} 分")

    return pd.DataFrame(rows)


def plot_curves(runs, targets, title, path):
    """
    左: 回数 vs 誤り率  右: 時間 vs 誤り率 (点線: 目標の誤り率)
    """

    colors = {"active": "#d62728", "random": "#7f7f7f"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    labeled = set()
    for strategy, seed, r in runs:
        label = strategy if strategy not in labeled else None
        labeled.add(strategy)

        axes[0].plot(r["n_labeled"], r["test_err"], color=colors[strategy],
                     marker="o", ms=2, lw=1, alpha=0.8, label=label)
        axes[1].plot(r["time_s"] / 60, r["test_err"], color=colors[strategy],
                     marker="o", ms=2, lw=1, alpha=0.8, label=label)

    for ax in axes:
        for t in targets:
            ax.axhline(1 - t, color="k", ls=":", lw=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    axes[0].set_xlabel("シミュレーション回数")
    axes[0].set_ylabel("既存グリッドでの誤り率")
    axes[0].set_title(f"{title}: 回数 vs 誤り率 (点線: 目標)")
    axes[1].set_xlabel("時間 [分] (random は シミュレーション合計 + 最後の学習)")
    axes[1].set_title(f"{title}: 時間 vs 誤り率")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare(circuit):

    print(f"\n===== {circuit} =====")

    runs = find_runs(CIRCUITS[circuit])
    if not runs:
        print("  結果が見つかりません")
        return

    out_dir = os.path.join(STUDY_DIR, f"{circuit}_active_vs_random")
    os.makedirs(out_dir, exist_ok=True)

    targets = make_targets(runs)
    reach = reach_table(runs, targets)
    reach.to_csv(os.path.join(out_dir, "reach.csv"), index=False)

    plot_curves(runs, targets, circuit, os.path.join(out_dir, "curves.png"))

    # 方法・目標ごとの平均 (届いた実行だけで平均。reached = 届いた実行の数)
    summary = reach.groupby(["target", "strategy"]).agg(
        runs=("seed", "size"),
        reached=("n_sim", "count"),
        n_sim=("n_sim", "mean"),
        time_min=("time_min", "mean"),
    ).round(1)

    pd.set_option("display.width", 200)
    print("\n  目標の accuracy に届いた回数と時間 (届いた実行の平均):")
    print(summary.to_string())

    # active に対する random の倍率
    wide = summary[["n_sim", "time_min"]].unstack("strategy")
    if ("n_sim", "random") in wide and ("n_sim", "active") in wide:
        ratio = pd.DataFrame({
            "n_sim の比": wide[("n_sim", "random")] / wide[("n_sim", "active")],
            "時間の比": wide[("time_min", "random")] / wide[("time_min", "active")],
        }).round(1)
        print("\n  random ÷ active:")
        print(ratio.to_string())

    print(f"\n  保存先: {out_dir}")


def main():

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    circuits = list(CIRCUITS) if arg == "all" else [arg]

    for c in circuits:
        if c not in CIRCUITS:
            print(f"回路は {list(CIRCUITS)} か all: {c}")
            return
        compare(c)


if __name__ == "__main__":
    main()
