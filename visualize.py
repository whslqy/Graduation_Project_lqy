import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_result_label(result):
    if "variant_label" in result:
        return result["variant_label"]
    if "teacher_name" in result and "student_name" in result:
        return f"{result['teacher_name']} -> {result['student_name']}"
    return result["model_name"]


def slugify(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("->", "_to_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def save_figure(fig, save_path=None, close=False):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    if close:
        plt.close(fig)


def style_axes(axes):
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def compact_width(num_items, base=6.8, per_item=0.45, max_width=9.0):
    return min(max_width, max(base, num_items * per_item))


def compact_title(title, max_len=26):
    return title if len(title) <= max_len else f"{title[:max_len - 3]}..."


def collect_predictions(model, dataloader, device, forward_fn=None):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if forward_fn is None:
                logits = model(images)
            else:
                logits = forward_fn(model, images)

            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return np.array(all_labels), np.array(all_predictions)


def compute_confusion_matrix(labels, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label, pred_label] += 1
    return matrix


def plot_training_history(history, title_prefix="Training"):
    plt.style.use("classic")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    axes[0].plot(epochs, history.get("train_loss", []), label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)

    axes[1].plot(epochs, history.get("train_accuracy", []), label="Train Acc")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history.get("val_accuracy", []), label="Val Acc")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    style_axes(axes)

    plt.tight_layout()
    return fig, axes


def plot_distillation_history(history, title_prefix="Distillation"):
    plt.style.use("classic")
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)

    axes[0, 0].plot(epochs, history.get("train_loss", []), label="Train Loss")
    axes[0, 0].plot(epochs, history.get("val_loss", []), label="Val Loss")
    axes[0, 0].set_title(f"{title_prefix} Total Loss")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=True)

    axes[0, 1].plot(epochs, history.get("train_accuracy", []), label="Train Acc")
    axes[0, 1].plot(epochs, history.get("val_accuracy", []), label="Val Acc")
    axes[0, 1].set_title(f"{title_prefix} Accuracy")
    axes[0, 1].legend(loc="lower right", fontsize=8, frameon=True)

    axes[1, 0].plot(epochs, history.get("train_ce_loss", []), label="Train CE")
    axes[1, 0].plot(epochs, history.get("train_kd_loss", []), label="Train KD")
    axes[1, 0].plot(epochs, history.get("train_feature_loss", []), label="Train Feature")
    axes[1, 0].set_title(f"{title_prefix} Train Loss Components")
    axes[1, 0].legend(loc="lower left", fontsize=8, frameon=True)

    axes[1, 1].plot(epochs, history.get("val_ce_loss", []), label="Val CE")
    axes[1, 1].plot(epochs, history.get("val_kd_loss", []), label="Val KD")
    axes[1, 1].plot(epochs, history.get("val_feature_loss", []), label="Val Feature")
    axes[1, 1].set_title(f"{title_prefix} Val Loss Components")
    axes[1, 1].legend(loc="lower right", fontsize=8, frameon=True)
    for axis_row in axes:
        for axis in axis_row:
            axis.set_xlabel("Epoch")
    style_axes(axes)

    plt.tight_layout()
    return fig, axes


def plot_confusion_matrix(matrix, class_names, title="Confusion Matrix", normalize=False):
    plt.style.use("classic")
    display_matrix = matrix.astype(np.float64)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display_matrix = display_matrix / row_sums

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = display_matrix.max() / 2 if display_matrix.size else 0
    for row in range(display_matrix.shape[0]):
        for col in range(display_matrix.shape[1]):
            value = display_matrix[row, col]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    plt.tight_layout()
    return fig, ax


def plot_confusion_matrix_comparison(matrices, class_names, titles, normalize=False):
    plt.style.use("classic")
    num_plots = len(matrices)
    cols = 2
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        row, col = divmod(idx, cols)
        display_matrix = matrix.astype(np.float64)
        if normalize:
            row_sums = display_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            display_matrix = display_matrix / row_sums

        ax = axes[row, col]
        image = ax.imshow(display_matrix, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        threshold = display_matrix.max() / 2 if display_matrix.size else 0
        for r in range(display_matrix.shape[0]):
            for c in range(display_matrix.shape[1]):
                value = display_matrix[r, c]
                text = f"{value:.2f}" if normalize else f"{int(value)}"
                ax.text(c, r, text, ha="center", va="center", color="white" if value > threshold else "black")

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(num_plots, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_metric_comparison(results, train_key, val_key, labels, title):
    plt.style.use("classic")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    title = compact_title(title, max_len=24)

    for result, label in zip(results, labels):
        history = result["history"]
        epochs = np.arange(1, len(history.get(train_key, [])) + 1)
        axes[0].plot(epochs, history.get(train_key, []), label=label)
        axes[1].plot(epochs, history.get(val_key, []), label=label)

    axes[0].set_title(f"{title} Train", fontsize=9, pad=12)
    axes[0].set_xlabel("Epoch")
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)

    axes[1].set_title(f"{title} Val", fontsize=9, pad=12)
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    style_axes(axes)

    plt.tight_layout()
    return fig, axes


def plot_result_metric_bars(labels, values, title, ylabel="Accuracy", color="steelblue"):
    plt.style.use("classic")
    fig, ax = plt.subplots(figsize=(compact_width(len(labels), base=6.4, per_item=0.5, max_width=8.6), 4.2))
    x = np.arange(len(labels))
    ax.plot(x, values, color=color, marker="o", linewidth=2.0, markersize=6)
    ax.set_title(title, fontsize=9, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    style_axes([ax])

    value_offset = max(values) * 0.012 if values else 0.01
    top_padding = max(values) * 0.12 if values else 0.05
    if values:
        ax.set_ylim(min(values) - top_padding * 0.35, max(values) + top_padding)
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + value_offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig, ax


def plot_distillation_gains(labels, baseline_values, distilled_values, title="Distillation Gain"):
    plt.style.use("classic")
    gains = np.array(distilled_values) - np.array(baseline_values)
    line_colors = ["seagreen" if value >= 0 else "indianred" for value in gains]

    fig, ax = plt.subplots(figsize=(compact_width(len(labels), base=6.4, per_item=0.5, max_width=8.6), 4.2))
    x = np.arange(len(labels))
    ax.plot(x, gains, color="#355c7d", marker="o", linewidth=2.0, markersize=6)
    ax.scatter(x, gains, c=line_colors, s=36, zorder=3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(title, fontsize=9, pad=14)
    ax.set_ylabel("Accuracy Gain")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    style_axes([ax])

    if len(gains) > 0:
        spread = max(abs(gains.min()), abs(gains.max()), 0.01)
        ax.set_ylim(gains.min() - spread * 0.35, gains.max() + spread * 0.35)
    for idx, value in enumerate(gains):
        ax.text(
            idx,
            value,
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    plt.tight_layout()
    return fig, ax


def visualize_baseline_result(result, class_names, normalize_confusion=False):
    fig_history, _ = plot_training_history(result["history"], title_prefix=result["model_name"])
    fig_confusion, _ = plot_confusion_matrix(
        result["confusion_matrix"],
        class_names=class_names,
        title=f"{result['model_name']} Confusion Matrix",
        normalize=normalize_confusion,
    )
    plt.show()
    return {
        "history_figure": fig_history,
        "confusion_figure": fig_confusion,
    }


def visualize_distillation_result(result, class_names, normalize_confusion=False):
    title_prefix = f"{result['teacher_name']} -> {result['student_name']}"
    fig_history, _ = plot_distillation_history(result["history"], title_prefix=title_prefix)
    fig_confusion, _ = plot_confusion_matrix(
        result["confusion_matrix"],
        class_names=class_names,
        title=f"{title_prefix} Confusion Matrix",
        normalize=normalize_confusion,
    )
    plt.show()
    return {
        "history_figure": fig_history,
        "confusion_figure": fig_confusion,
    }


def compare_confusion_matrices(results, class_names, normalize=False):
    matrices = []
    titles = []
    for result in results:
        matrices.append(result["confusion_matrix"])
        titles.append(get_result_label(result))
    fig, _ = plot_confusion_matrix_comparison(matrices, class_names, titles, normalize=normalize)
    plt.show()
    return fig


def compare_training_curves(results, train_key="train_accuracy", val_key="val_accuracy", title="Accuracy Comparison"):
    labels = [get_result_label(result) for result in results]
    fig, _ = plot_metric_comparison(results, train_key, val_key, labels, title)
    plt.show()
    return fig


def compare_student_baseline_and_distilled(baseline_results, distillation_runs, class_names, title="Student Comparison"):
    combined_results = list(baseline_results) + [item["student_result"] for item in distillation_runs]
    compare_training_curves(combined_results, title=title)
    compare_confusion_matrices(combined_results, class_names=class_names, normalize=False)


def visualize_distillation_gains_summary(baseline_results, distillation_runs, metric_key="test_accuracy"):
    baseline_map = {result["model_name"]: result for result in baseline_results}
    labels = []
    baseline_values = []
    distilled_values = []

    for run in distillation_runs:
        student_result = run["student_result"]
        student_name = student_result["student_name"]
        if student_name not in baseline_map:
            continue

        if metric_key == "best_val_accuracy":
            baseline_value = baseline_map[student_name]["history"]["best_val_accuracy"]
            distilled_value = student_result["history"]["best_val_accuracy"]
        else:
            baseline_value = baseline_map[student_name]["test_metrics"]["accuracy"]
            distilled_value = student_result["test_metrics"]["accuracy"]

        labels.append(f"{run['teacher_name']} -> {student_name}")
        baseline_values.append(baseline_value)
        distilled_values.append(distilled_value)

    fig_bars, _ = plot_result_metric_bars(labels, distilled_values, title=f"{metric_key} of Distilled Students")
    plt.show()
    fig_gains, _ = plot_distillation_gains(labels, baseline_values, distilled_values, title=f"Distillation Gain ({metric_key})")
    plt.show()
    return {
        "bars_figure": fig_bars,
        "gains_figure": fig_gains,
    }
