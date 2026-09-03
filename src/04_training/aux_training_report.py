import argparse
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_professional_report(target_dir: str, output_file: str):
    base_path = Path(target_dir)

    all_state_files = list(base_path.rglob("trainer_state.json"))

    if not all_state_files:
        if (base_path / "trainer_state.json").exists():
            all_state_files = [base_path / "trainer_state.json"]
        else:
            logger.error(f"Could not find any 'trainer_state.json' in {target_dir} or its subdirectories.")
            return

    logger.info(f"Scanning {len(all_state_files)} state files to build the complete training trajectory...")

    full_log_history = []
    for state_file in all_state_files:
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                history = state_data.get("log_history", [])
                if len(history) > len(full_log_history):
                    full_log_history = history
        except Exception:
            pass

    if not full_log_history:
        logger.error("The log_history found is empty.")
        return

    train_logs = [log for log in full_log_history if "loss" in log and "eval_loss" not in log]
    eval_logs = [log for log in full_log_history if "eval_loss" in log]

    if not train_logs or not eval_logs:
        logger.warning("Missing either training or evaluation logs. Graph may be incomplete.")

    df_train = pd.DataFrame(train_logs)
    df_eval = pd.DataFrame(eval_logs)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            "figure.autolayout": True,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    model_name = base_path.name
    fig.suptitle(f"Full Training Convergence Report: {model_name}", fontweight="bold", y=0.98)

    if not df_train.empty:
        sns.lineplot(
            data=df_train, x="step", y="loss", ax=ax1, label="Training Loss", color="#2c3e50", linewidth=1.5, alpha=0.8
        )

    if not df_eval.empty:
        sns.lineplot(
            data=df_eval,
            x="step",
            y="eval_loss",
            ax=ax1,
            label="Validation Loss",
            color="#e74c3c",
            marker="o",
            markersize=6,
            linewidth=2,
        )

        min_eval = df_eval.loc[df_eval["eval_loss"].idxmin()]
        ax1.axvline(min_eval["step"], color="#27ae60", linestyle="--", alpha=0.7, zorder=0)
        ax1.annotate(
            f"Convergence Point\nBest Eval: {min_eval['eval_loss']:.4f}\nStep: {int(min_eval['step'])}",
            xy=(min_eval["step"], min_eval["eval_loss"]),
            xytext=(15, 25),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#27ae60"),
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="#27ae60", facecolor="#eafaf1", alpha=0.9),
        )

    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Model Convergence Trajectory (Full Run)")
    ax1.legend(loc="upper right", frameon=True, shadow=True)

    if not df_train.empty and "learning_rate" in df_train.columns:
        sns.lineplot(data=df_train, x="step", y="learning_rate", ax=ax2, color="#2980b9", linewidth=2)
        ax2.fill_between(df_train["step"], df_train["learning_rate"], alpha=0.1, color="#2980b9")

    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Global Training Steps")
    ax2.set_title("Learning Rate Schedule")

    if not df_train.empty and "epoch" in df_train.columns:
        ax2_epochs = ax2.twiny()
        ax2_epochs.set_xlim(ax2.get_xlim())

        step_to_epoch = df_train[["step", "epoch"]].dropna().drop_duplicates(subset=["epoch"], keep="first")
        epoch_ticks = step_to_epoch["step"].values
        epoch_labels = [f"Ep {int(e)}" if e.is_integer() else f"Ep {e:.1f}" for e in step_to_epoch["epoch"].values]

        if len(epoch_ticks) > 10:
            epoch_ticks = epoch_ticks[:: len(epoch_ticks) // 10]
            epoch_labels = epoch_labels[:: len(epoch_labels) // 10]

        ax2_epochs.set_xticks(epoch_ticks)
        ax2_epochs.set_xticklabels(epoch_labels, rotation=45, ha="left", fontsize=9)
        ax2_epochs.set_xlabel("Epochs", labelpad=10)
        ax2_epochs.grid(False)

    sns.despine(left=True, bottom=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Professional report successfully generated: {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-quality full training graphs.")
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to the main checkpoints folder (e.g., adapters/checkpoints_r16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="full_training_report.png",
        help="Name of the output image file (default: full_training_report.png)",
    )

    args = parser.parse_args()
    generate_professional_report(args.target_dir, args.output)
