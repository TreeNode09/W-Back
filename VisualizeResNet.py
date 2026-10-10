"""Plot ResNet18 training curves from embedded PRC / Tree-Ring histories."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

# Save under Data/exp when present; otherwise cwd.
_OUT_DIRS = [
    os.path.join("Data", "exp"),
    os.path.join("exp"),
    ".",
]

MAX_EPOCH = 10
FONT_SIZE = 20
DPI = 300
Y_TICK_STEP = 10.0
Y_TICK_PAD = 0.2  # margin outside end ticks, as a fraction of Y_TICK_STEP

# key: "loss" | "train_acc" | "val_acc"
# PLOT = ("loss", "resnet_loss_epoch10.png")
# PLOT = ("train_acc", "resnet_train_acc_epoch10.png")
PLOT = ("val_acc", "resnet_val_acc_epoch10.png")

_YLABELS = {
    "loss": "损失",
    "train_acc": "准确率(%)",
    "val_acc": "准确率(%)",
}

PRC = [
    {"epoch": 1.0, "loss": 0.727872209072113, "train_acc": 0.755, "val_acc": 0.5155},
    {"epoch": 2.0, "loss": 0.5015849752426147, "train_acc": 0.89825, "val_acc": 0.5095},
    {"epoch": 3.0, "loss": 0.17910103225708007, "train_acc": 0.970375, "val_acc": 0.4985},
    {"epoch": 4.0, "loss": 0.03815189679712057, "train_acc": 0.999875, "val_acc": 0.4835},
    {"epoch": 5.0, "loss": 0.009036131277680397, "train_acc": 1.0, "val_acc": 0.5065},
    {"epoch": 6.0, "loss": 0.0038491715267300604, "train_acc": 1.0, "val_acc": 0.4915},
    {"epoch": 7.0, "loss": 0.0018014921890571714, "train_acc": 1.0, "val_acc": 0.489},
    {"epoch": 8.0, "loss": 0.0012969355357345194, "train_acc": 1.0, "val_acc": 0.496},
    {"epoch": 9.0, "loss": 0.0008467949291225522, "train_acc": 1.0, "val_acc": 0.4995},
    {"epoch": 10.0, "loss": 0.0006535322088748217, "train_acc": 1.0, "val_acc": 0.496},
    {"epoch": 11.0, "loss": 0.0005149170029908419, "train_acc": 1.0, "val_acc": 0.4985},
    {"epoch": 12.0, "loss": 0.00048065414885059, "train_acc": 1.0, "val_acc": 0.493},
    {"epoch": 13.0, "loss": 0.0004504585327813402, "train_acc": 1.0, "val_acc": 0.4955},
    {"epoch": 14.0, "loss": 0.000354522813053336, "train_acc": 1.0, "val_acc": 0.502},
    {"epoch": 15.0, "loss": 0.00039380127558251844, "train_acc": 1.0, "val_acc": 0.5025},
    {"epoch": 16.0, "loss": 0.0002423252885346301, "train_acc": 1.0, "val_acc": 0.496},
    {"epoch": 17.0, "loss": 0.00024337393569294363, "train_acc": 1.0, "val_acc": 0.499},
    {"epoch": 18.0, "loss": 0.00022538295958656818, "train_acc": 1.0, "val_acc": 0.4945},
    {"epoch": 19.0, "loss": 0.0001590839424461592, "train_acc": 1.0, "val_acc": 0.494},
    {"epoch": 20.0, "loss": 0.00012976953125325963, "train_acc": 1.0, "val_acc": 0.4965},
    {"epoch": 21.0, "loss": 0.00013793937070295215, "train_acc": 1.0, "val_acc": 0.4955},
    {"epoch": 22.0, "loss": 0.0001152296902728267, "train_acc": 1.0, "val_acc": 0.492},
    {"epoch": 23.0, "loss": 8.347253885585814e-05, "train_acc": 1.0, "val_acc": 0.499},
    {"epoch": 24.0, "loss": 0.00010830402925785165, "train_acc": 1.0, "val_acc": 0.4945},
    {"epoch": 25.0, "loss": 9.975450145429932e-05, "train_acc": 1.0, "val_acc": 0.495},
    {"epoch": 26.0, "loss": 0.00011169560125563294, "train_acc": 1.0, "val_acc": 0.49},
    {"epoch": 27.0, "loss": 8.224975182383786e-05, "train_acc": 1.0, "val_acc": 0.495},
    {"epoch": 28.0, "loss": 6.349653685174416e-05, "train_acc": 1.0, "val_acc": 0.498},
    {"epoch": 29.0, "loss": 6.693403013923671e-05, "train_acc": 1.0, "val_acc": 0.489},
    {"epoch": 30.0, "loss": 6.736127036128892e-05, "train_acc": 1.0, "val_acc": 0.491},
]

TREE_RING = [
    {"epoch": 1.0, "loss": 0.3700591086149216, "train_acc": 0.9805, "val_acc": 0.9225},
    {"epoch": 2.0, "loss": 0.06830900700390338, "train_acc": 0.996875, "val_acc": 0.9395},
    {"epoch": 3.0, "loss": 0.017196358576416968, "train_acc": 0.99975, "val_acc": 0.957},
    {"epoch": 4.0, "loss": 0.0037073037200607358, "train_acc": 1.0, "val_acc": 0.9625},
    {"epoch": 5.0, "loss": 0.001493206751300022, "train_acc": 1.0, "val_acc": 0.962},
    {"epoch": 6.0, "loss": 0.0011537394982296973, "train_acc": 1.0, "val_acc": 0.964},
    {"epoch": 7.0, "loss": 0.0006748394012684002, "train_acc": 1.0, "val_acc": 0.963},
    {"epoch": 8.0, "loss": 0.000594325622485485, "train_acc": 1.0, "val_acc": 0.9645},
    {"epoch": 9.0, "loss": 0.00043571593193337323, "train_acc": 1.0, "val_acc": 0.9625},
    {"epoch": 10.0, "loss": 0.00030931904172757643, "train_acc": 1.0, "val_acc": 0.965},
    {"epoch": 11.0, "loss": 0.000252567017218098, "train_acc": 1.0, "val_acc": 0.9645},
    {"epoch": 12.0, "loss": 0.0002637135038967244, "train_acc": 1.0, "val_acc": 0.9665},
    {"epoch": 13.0, "loss": 0.00024079091628664173, "train_acc": 1.0, "val_acc": 0.963},
    {"epoch": 14.0, "loss": 0.00022288510686485096, "train_acc": 1.0, "val_acc": 0.964},
    {"epoch": 15.0, "loss": 0.0001620943183952477, "train_acc": 1.0, "val_acc": 0.9635},
    {"epoch": 16.0, "loss": 0.00013290629102266394, "train_acc": 1.0, "val_acc": 0.964},
    {"epoch": 17.0, "loss": 0.00014532195062201935, "train_acc": 1.0, "val_acc": 0.963},
    {"epoch": 18.0, "loss": 0.00016489599882334006, "train_acc": 1.0, "val_acc": 0.964},
    {"epoch": 19.0, "loss": 0.00010358927152992692, "train_acc": 1.0, "val_acc": 0.9645},
    {"epoch": 20.0, "loss": 8.243209069769364e-05, "train_acc": 1.0, "val_acc": 0.966},
    {"epoch": 21.0, "loss": 8.701859260327183e-05, "train_acc": 1.0, "val_acc": 0.9695},
    {"epoch": 22.0, "loss": 7.918039979995228e-05, "train_acc": 1.0, "val_acc": 0.968},
    {"epoch": 23.0, "loss": 4.8863821597478815e-05, "train_acc": 1.0, "val_acc": 0.967},
    {"epoch": 24.0, "loss": 7.675327101605945e-05, "train_acc": 1.0, "val_acc": 0.9705},
    {"epoch": 25.0, "loss": 8.033420536230551e-05, "train_acc": 1.0, "val_acc": 0.9595},
    {"epoch": 26.0, "loss": 0.09684076151486806, "train_acc": 0.993, "val_acc": 0.979},
    {"epoch": 27.0, "loss": 0.02417766347154975, "train_acc": 0.999375, "val_acc": 0.985},
    {"epoch": 28.0, "loss": 0.00497894723364152, "train_acc": 1.0, "val_acc": 0.99},
    {"epoch": 29.0, "loss": 0.001561193546745926, "train_acc": 0.999875, "val_acc": 0.9905},
    {"epoch": 30.0, "loss": 0.0012338509180699474, "train_acc": 1.0, "val_acc": 0.9915},
]


def _series(rows: list[dict], key: str, max_epoch: int) -> tuple[list[float], list[float]]:

    xs, ys = [], []
    for row in rows:

        ep = int(row["epoch"])
        if ep > max_epoch: continue

        xs.append(ep)
        ys.append(float(row[key]))

    return xs, ys


def _yticks_covering(values: list[float], step: float) -> list[float]:
    """Ticks at k*step that fully cover data (data stays between first and last tick)."""

    import math

    if step <= 0:
        raise ValueError(f"Y_TICK_STEP={step} must be positive")
    if not values:
        return [0.0, step]

    lo_v = min(values)
    hi_v = max(values)
    n0 = math.floor(lo_v / step + 1e-12)
    n1 = math.ceil(hi_v / step - 1e-12)
    if n0 == n1:
        n0 -= 1
        n1 += 1
    return [i * step for i in range(n0, n1 + 1)]


def resolve_out_path(filename: str) -> str:

    for folder in _OUT_DIRS:

        if folder == "." or os.path.isdir(folder):  return os.path.join(folder, filename)

    return filename


def plot_metric(key: str | None = None,filename: str | None = None,
    max_epoch: int = MAX_EPOCH, out_path: str | None = None) -> str:

    from matplotlib import font_manager

    metric_key = key if key is not None else PLOT[0]
    out_name = filename if filename is not None else PLOT[1]
    if metric_key not in _YLABELS: raise ValueError(f"unknown metric key={metric_key!r}; expected one of {list(_YLABELS)}")

    out_path = out_path or resolve_out_path(out_name)
    parent = os.path.dirname(out_path)
    if parent: os.makedirs(parent, exist_ok=True)

    plt.rcParams["axes.unicode_minus"] = False
    font_cn = font_manager.FontProperties(family="SimHei", size=FONT_SIZE)
    font_en = font_manager.FontProperties(family="Times New Roman", size=FONT_SIZE)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=DPI)

    styles = (
        ("PRC", PRC, "#ED7D31"),
        ("Tree-Ring", TREE_RING, "#4472C4"),
    )

    all_ys: list[float] = []
    for label, rows, color in styles:

        xs, ys = _series(rows, metric_key, max_epoch)
        if metric_key in ["train_acc", "val_acc"]: ys = [y * 100 for y in ys]
        all_ys.extend(ys)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5, label=label, color=color)

    yticks = _yticks_covering(all_ys, Y_TICK_STEP)
    pad = Y_TICK_STEP * Y_TICK_PAD
    ax.set_ylim(yticks[0] - pad, yticks[-1] + pad)
    ax.set_yticks(yticks)

    ax.set_xlabel("轮次", fontproperties=font_cn)
    ax.set_ylabel(_YLABELS[metric_key], fontproperties=font_cn)
    ax.set_xticks(range(1, max_epoch + 1))
    ax.tick_params(labelsize=FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.4)

    for tick in ax.get_xticklabels() + ax.get_yticklabels(): tick.set_fontproperties(font_en)

    ax.legend(loc="center right", frameon=True, prop=font_en)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)

    return out_path


if __name__ == "__main__":

    path = plot_metric()
    print(f"saved {path}")
