import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize forecasts")
    parser.add_argument("--run_dir", type=str, required=True, help="Folder with input.npy, pred.npy, true.npy, metadata.json")
    parser.add_argument("--sample_idx", type=int, default=0, help="First test-window index to plot")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of forecast windows to place into one PNG")
    parser.add_argument(
        "--sample_step",
        type=int,
        default=None,
        help="Distance between automatically selected sample indices. Default pred_len.",
    )
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None, help="Explicit sample indices to plot")
    parser.add_argument("--channel_idx", type=int, default=None, help="Feature index to plot")
    parser.add_argument("--channel_name", type=str, default=None, help="Feature name to plot")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the forecast grid")
    parser.add_argument("--output", type=str, default=None, help="Output image path (.png)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--hide_history", action="store_true", default=False)
    return parser


def resolve_channel(metadata, channel_idx, channel_name):
    feature_names = metadata.get("feature_names", [])

    if channel_name is not None:
        return feature_names.index(channel_name), channel_name

    if channel_idx is None:
        if feature_names:
            return len(feature_names) - 1, feature_names[-1]
        return -1, "channel_-1"

    label = feature_names[channel_idx] if feature_names else f"channel_{channel_idx}"
    return channel_idx, label


def load_history(run_dir):
    history_path = os.path.join(run_dir, "history.json")
    if not os.path.exists(history_path):
        history_npy_path = os.path.join(run_dir, "history.npy")

        history_array = np.load(history_npy_path)
        return [
            {
                "epoch": int(row[0]),
                "train_loss": float(row[1]),
                "val_loss": float(row[2]),
                "test_loss": float(row[3]),
                "learning_rate": float(row[4]),
            }
            for row in history_array
        ]

    with open(history_path, "r", encoding="utf-8") as file:
        return json.load(file)

def maybe_inverse_saved_arrays(metadata, inputs, preds, trues):
    if metadata.get("saved_arrays_are_inverse_scaled", False):
        return inputs, preds, trues, False

    if not metadata.get("scale_applied", False):
        return inputs, preds, trues, False

    raw_feature_names = metadata.get("raw_feature_names") or metadata.get("feature_names") or []
    saved_feature_names = metadata.get("feature_names") or raw_feature_names
    scaler_mean = metadata.get("scaler_mean")
    scaler_std = metadata.get("scaler_std")

    if scaler_mean is None or scaler_std is None or not raw_feature_names or not saved_feature_names:
        return inputs, preds, trues, False

    try:
        channel_indices = [raw_feature_names.index(name) for name in saved_feature_names]
    except ValueError:
        return inputs, preds, trues, False

    mean = np.array(scaler_mean, dtype=np.float32)[channel_indices].reshape(1, 1, -1)
    std = np.array(scaler_std, dtype=np.float32)[channel_indices].reshape(1, 1, -1)

    def inverse_transform(array):
        return (array * std) + mean

    return inverse_transform(inputs), inverse_transform(preds), inverse_transform(trues), True


def compute_sample_metrics(true, pred):
    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))

    return {
        "MAE": mae,
        "RMSE": rmse,
    }


def format_metrics(metrics):
    return "\n".join(
        [
            f"MAE   = {metrics['MAE']:.4f}",
            f"RMSE  = {metrics['RMSE']:.4f}",
        ]
    )


def plot_history(ax, history, best_epoch=None):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]

    ax.plot(epochs, train_loss, label="Train loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val loss", linewidth=2)
    ax.plot(epochs, test_loss, label="Test loss", linewidth=2, linestyle="--")

    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.25, label=f"Best epoch = {best_epoch}")

    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()


def resolve_sample_indices(total_samples, sample_idx, num_samples, sample_indices, sample_step):
    if sample_indices is not None and len(sample_indices) > 0:
        resolved = sample_indices
    else:
        step = max(1, int(sample_step))
        resolved = []
        current_index = sample_idx
        while len(resolved) < num_samples and current_index < total_samples:
            resolved.append(current_index)
            current_index += step

    return resolved


def get_overall_metrics(metadata):
    if "metrics_inverse" in metadata:
        metrics = metadata["metrics_inverse"]
        scale_label = "original scale"
    else:
        metrics = metadata.get("metrics", {})

        if metadata.get("metrics_are_inverse_scaled", False):
            scale_label = "original scale"
        elif metadata.get("scaled_metrics", False):
            scale_label = "standardized scale"
        else:
            scale_label = "original scale"

    ordered_keys = ["mae", "rmse"]
    parts = []
    for key in ordered_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        parts.append(f"{key.upper()}={value:.4f}")
    return (" | ".join(parts) if parts else None), scale_label


def plot_sample(ax, sample_index, channel_label, input_history, true, pred):
    x_history = np.arange(len(input_history))
    x_forecast = np.arange(len(input_history), len(input_history) + len(pred))
    sample_metrics = compute_sample_metrics(true, pred)

    ax.plot(x_history, input_history, label="History", linewidth=1.8)
    ax.plot(x_forecast, true, label="Ground truth", linewidth=1.8)
    ax.plot(x_forecast, pred, label="Prediction", linewidth=1.8)
    ax.axvline(len(input_history) - 1, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Sample {sample_index} | {channel_label}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.text(
        0.015,
        0.98,
        format_metrics(sample_metrics),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="lightgray"),
    )


def main():
    args = build_parser().parse_args()

    metadata_path = os.path.join(args.run_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    inputs = np.load(os.path.join(args.run_dir, "input.npy"))
    preds = np.load(os.path.join(args.run_dir, "pred.npy"))
    trues = np.load(os.path.join(args.run_dir, "true.npy"))
    inputs, preds, trues, auto_inversed = maybe_inverse_saved_arrays(metadata, inputs, preds, trues)
    training_history = [] if args.hide_history else load_history(args.run_dir)
    sample_step = args.sample_step if args.sample_step is not None else metadata.get("pred_len", 1)
    sample_indices = resolve_sample_indices(
        len(preds),
        args.sample_idx,
        args.num_samples,
        args.sample_indices,
        sample_step,
    )

    channel_idx, channel_label = resolve_channel(metadata, args.channel_idx, args.channel_name)
    cols = max(1, args.cols)
    sample_rows = math.ceil(len(sample_indices) / cols)
    overall_metrics_line, metrics_scale_label = get_overall_metrics(metadata)

    if training_history:
        fig = plt.figure(figsize=(6.2 * cols, 3.6 * (sample_rows + 1.1)))
        grid = fig.add_gridspec(sample_rows + 1, cols, height_ratios=[1.2] + [1.0] * sample_rows)
        ax_history = fig.add_subplot(grid[0, :])
        plot_history(ax_history, training_history, best_epoch=metadata.get("best_epoch"))
        history_scale = metadata.get("train_history_loss_scale")
        if history_scale:
            ax_history.set_title(f"Training History ({history_scale} loss)")
        sample_axes = [fig.add_subplot(grid[row + 1, col]) for row in range(sample_rows) for col in range(cols)]
    else:
        fig, axes = plt.subplots(sample_rows, cols, figsize=(6.2 * cols, 3.6 * sample_rows))
        axes = np.array(axes).reshape(-1)
        sample_axes = list(axes)
        ax_history = None

    for axis, sample_index in zip(sample_axes, sample_indices):
        input_history = inputs[sample_index, :, channel_idx]
        pred = preds[sample_index, :, channel_idx]
        true = trues[sample_index, :, channel_idx]
        plot_sample(axis, sample_index, channel_label, input_history, true, pred)

    for axis in sample_axes[len(sample_indices):]:
        axis.axis("off")

    if sample_axes:
        sample_axes[0].legend(loc="upper right")

    value_scale_label = "original scale"
    if auto_inversed:
        value_scale_label += " (auto-restored from scaler)"

    title = f"Forecasts | channel={channel_label} | samples={sample_indices} | values={value_scale_label}"
    if overall_metrics_line is not None:
        label_suffix = f" [{metrics_scale_label}]" if metrics_scale_label is not None else ""
        title += f"\nTest metrics{label_suffix}: {overall_metrics_line}"
    fig.suptitle(title, y=0.995, fontsize=14)

    output_path = args.output
    if output_path is None:
        safe_label = str(channel_label).replace(" ", "_")
        if len(sample_indices) == 1:
            output_path = os.path.join(args.run_dir, f"forecast_sample_{sample_indices[0]}_{safe_label}.png")
        else:
            output_path = os.path.join(
                args.run_dir,
                f"forecast_grid_{sample_indices[0]}_{sample_indices[-1]}_{safe_label}.png",
            )

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved plot to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
