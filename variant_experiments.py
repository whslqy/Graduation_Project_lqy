import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from kd_config import CLASS_NAMES, DISTILLATION_CONFIG
from kd_models import normalize_model_name
from pipeline import prepare_context, run_student_baselines, train_distilled_model, train_teacher_model
from visualize import (
    compare_confusion_matrices,
    compare_training_curves,
    plot_distillation_gains,
    plot_result_metric_bars,
    save_figure,
    slugify,
    visualize_baseline_result,
)


def history_to_dataframe(history):
    if not history:
        return pd.DataFrame()

    sequence_lengths = []
    for value in history.values():
        if isinstance(value, list):
            sequence_lengths.append(len(value))

    if not sequence_lengths:
        return pd.DataFrame()

    num_epochs = max(sequence_lengths)
    frame = {"epoch": list(range(1, num_epochs + 1))}
    for key, value in history.items():
        if isinstance(value, list):
            padded = list(value) + [None] * (num_epochs - len(value))
            frame[key] = padded
        else:
            frame[key] = [value] * num_epochs
    return pd.DataFrame(frame)


def export_history_csv(history, save_path, metadata=None):
    history_df = history_to_dataframe(history)
    if metadata:
        for key, value in metadata.items():
            history_df[key] = value
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return history_df


DISTILLATION_VARIANT_LIBRARY = {
    "logits_only": {
        "label": "Logits Only",
        "use_feature_distill": False,
        "feature_loss_type": None,
        "feature_loss_weight": 0.0,
    },
    "logits_cosine": {
        "label": "Logits + Cosine",
        "use_feature_distill": True,
        "feature_loss_type": "cosine",
        "feature_loss_weight": {
            "resnet18": 0.08,
            "mobilenet_v2": 0.03,
            "shufflenet_v2_x0_5": 0.015,
        },
    },
    "logits_mse_regressor": {
        "label": "Logits + Regressor MSE",
        "use_feature_distill": True,
        "feature_loss_type": "mse_regressor",
        "feature_loss_weight": {
            "resnet18": 0.05,
            "mobilenet_v2": 0.03,
            "shufflenet_v2_x0_5": 0.02,
        },
    },
    "logits_mse_regressor_cosine": {
        "label": "Logits + Regressor MSE + Cosine",
        "use_feature_distill": True,
        "feature_loss_type": "mse_regressor_cosine",
        "feature_loss_weight": {
            "resnet18": 0.03,
            "mobilenet_v2": 0.02,
            "shufflenet_v2_x0_5": 0.01,
        },
    },
}


SWEEP_TYPE_LIBRARY = {
    "logits_only": {
        "temperature": [2.0, 3.0, 4.0, 5.0],
        "alpha": [0.2, 0.35, 0.5, 0.65],
        "warmup_epochs": [3, 5, 8, 10],
        "kd_ramp_epochs": [3, 5, 8, 10],
    },
    "logits_cosine": {
        "temperature": [2.0, 3.0, 4.0, 5.0],
        "alpha": [0.2, 0.35, 0.5, 0.65],
        "feature_loss_weight": [0.01, 0.03, 0.05, 0.08],
        "feature_start_epoch": [5, 10, 15, 20],
    },
    "logits_mse_regressor": {
        "temperature": [2.0, 3.0, 4.0, 5.0],
        "alpha": [0.2, 0.35, 0.5, 0.65],
        "feature_loss_weight": [0.01, 0.02, 0.03, 0.05],
        "feature_start_epoch": [5, 10, 15, 20],
    },
    "logits_mse_regressor_cosine": {
        "temperature": [2.0, 3.0, 4.0, 5.0],
        "alpha": [0.15, 0.25, 0.35, 0.45],
        "feature_loss_weight": [0.005, 0.01, 0.02, 0.03],
        "feature_start_epoch": [8, 12, 16, 20],
    },
}


def build_variant_distillation_config(student_name, variant_name):
    variant = DISTILLATION_VARIANT_LIBRARY[variant_name]
    if not variant["use_feature_distill"]:
        return {
            "student_specific_config": {
                student_name: {
                    "feature_loss_weight": 0.0,
                }
            }
        }

    weight = variant["feature_loss_weight"]
    if isinstance(weight, dict):
        weight = weight.get(student_name, 0.02)

    return {
        "student_specific_config": {
            student_name: {
                "feature_loss_type": variant["feature_loss_type"],
                "feature_loss_weight": weight,
            }
        }
    }


def merge_distillation_configs(base_config, override_config):
    merged = {}
    for key, value in (base_config or {}).items():
        if isinstance(value, dict):
            merged[key] = dict(value)
        else:
            merged[key] = value

    for key, value in (override_config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict) and isinstance(nested.get(nested_key), dict):
                    deeper = dict(nested[nested_key])
                    deeper.update(nested_value)
                    nested[nested_key] = deeper
                else:
                    nested[nested_key] = nested_value
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def short_name(model_name):
    mapping = {
        "resnet50": "R50",
        "resnet18": "R18",
        "mobilenet_v2": "MBV2",
        "shufflenet_v2_x0_5": "SFV2",
        "inception_resnet_v2": "IRV2",
    }
    return mapping.get(model_name, model_name)


def run_distillation_variant_suite(
    teacher_name,
    student_name,
    baseline_results,
    teacher_result=None,
    teacher_pretrained=True,
    student_pretrained=True,
    teacher_training_config=None,
    student_training_config=None,
    teacher_save_path=None,
    student_save_dir=None,
    variant_names=None,
    show_augments=True,
    context=None,
):
    teacher_name = normalize_model_name(teacher_name)
    student_name = normalize_model_name(student_name)
    variant_names = variant_names or list(DISTILLATION_VARIANT_LIBRARY.keys())
    context = prepare_context(show_augments=show_augments) if context is None else context

    if teacher_result is None:
        teacher_result = train_teacher_model(
            model_name=teacher_name,
            pretrained=teacher_pretrained,
            save_path=teacher_save_path,
            training_config=teacher_training_config,
            context=context,
        )

    baseline_map = {result["model_name"]: result for result in baseline_results}
    if student_name not in baseline_map:
        raise ValueError(f"Missing baseline result for student model: {student_name}")

    baseline_result = baseline_map[student_name]
    variant_results = []
    for variant_name in variant_names:
        variant = DISTILLATION_VARIANT_LIBRARY[variant_name]
        student_save_path = None
        if student_save_dir is not None:
            student_save_path = f"{student_save_dir}/{teacher_name}_to_{student_name}_{variant_name}.pt"

        student_result = train_distilled_model(
            teacher_name=teacher_name,
            student_name=student_name,
            teacher_model=teacher_result["model"],
            pretrained_teacher=False,
            pretrained_student=student_pretrained,
            training_config=student_training_config,
            distillation_config=build_variant_distillation_config(student_name, variant_name),
            save_path=student_save_path,
            use_feature_distill=variant["use_feature_distill"],
            context=context,
            show_augments=False,
        )
        student_result["variant_name"] = variant_name
        student_result["variant_label"] = variant["label"]
        variant_results.append(student_result)

    return {
        "teacher_name": teacher_name,
        "student_name": student_name,
        "teacher_result": teacher_result,
        "baseline_result": baseline_result,
        "variant_results": variant_results,
    }


def run_all_distillation_variant_suites(
    baseline_results,
    experiments=None,
    teacher_pretrained=True,
    student_pretrained=True,
    teacher_training_config=None,
    student_training_config=None,
    teacher_save_dir=None,
    student_save_dir=None,
    variant_names=None,
    show_augments=True,
):
    experiments = experiments or DISTILLATION_CONFIG["experiments"]
    shared_context = prepare_context(show_augments=show_augments)
    trained_teachers = {}
    suites = []

    for experiment in experiments:
        teacher_name = normalize_model_name(experiment["teacher"])
        student_name = normalize_model_name(experiment["student"])

        if teacher_name not in trained_teachers:
            teacher_save_path = None
            if teacher_save_dir is not None:
                teacher_save_path = f"{teacher_save_dir}/{teacher_name}.pt"
            trained_teachers[teacher_name] = train_teacher_model(
                model_name=teacher_name,
                pretrained=teacher_pretrained,
                save_path=teacher_save_path,
                training_config=teacher_training_config,
                context=shared_context,
            )

        suite = run_distillation_variant_suite(
            teacher_name=teacher_name,
            student_name=student_name,
            baseline_results=baseline_results,
            teacher_result=trained_teachers[teacher_name],
            teacher_pretrained=teacher_pretrained,
            student_pretrained=student_pretrained,
            teacher_training_config=teacher_training_config,
            student_training_config=student_training_config,
            student_save_dir=student_save_dir,
            variant_names=variant_names,
            show_augments=False,
            context=shared_context,
        )
        suites.append(suite)

    return suites


def summarize_variant_suite(suite_result, metric_key="test_accuracy"):
    baseline_result = suite_result["baseline_result"]
    baseline_value = baseline_result["history"]["best_val_accuracy"] if metric_key == "best_val_accuracy" else baseline_result["test_metrics"]["accuracy"]

    records = [
        {
            "teacher": suite_result["teacher_name"],
            "student": suite_result["student_name"],
            "variant": "Baseline",
            "feature_loss_type": "none",
            "best_val_accuracy": baseline_result["history"]["best_val_accuracy"],
            "test_accuracy": baseline_result["test_metrics"]["accuracy"],
            "baseline_accuracy": baseline_value,
            "gain_vs_baseline": 0.0,
        }
    ]
    for result in suite_result["variant_results"]:
        metric_value = result["history"]["best_val_accuracy"] if metric_key == "best_val_accuracy" else result["test_metrics"]["accuracy"]
        records.append(
            {
                "teacher": suite_result["teacher_name"],
                "student": suite_result["student_name"],
                "variant": result["variant_label"],
                "feature_loss_type": result.get("feature_loss_type", "none"),
                "best_val_accuracy": result["history"]["best_val_accuracy"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "baseline_accuracy": baseline_value,
                "gain_vs_baseline": metric_value - baseline_value,
            }
        )
    return pd.DataFrame(records)


def visualize_variant_suite(suite_result, metric_key="test_accuracy"):
    summary_df = summarize_variant_suite(suite_result, metric_key=metric_key)
    labels = summary_df["variant"].tolist()
    baseline_values = summary_df["baseline_accuracy"].tolist()
    distilled_values = summary_df[metric_key].tolist()
    title = f"{suite_result['teacher_name']} -> {suite_result['student_name']} ({metric_key})"
    plot_distillation_gains(labels, baseline_values, distilled_values, title=title)
    plt.show()
    return summary_df


def display_variant_suite_tables(suite_results, metric_key="test_accuracy"):
    tables = {}
    for suite_result in suite_results:
        key = f"{suite_result['teacher_name']} -> {suite_result['student_name']}"
        print(f"\n=== {key} ===")
        summary_df = summarize_variant_suite(suite_result, metric_key=metric_key)
        display(summary_df)
        visualize_variant_suite(suite_result, metric_key=metric_key)
        tables[key] = summary_df
    return tables


def summarize_baseline_results(baseline_results):
    records = []
    for result in baseline_results:
        records.append(
            {
                "student": result["model_name"],
                "best_val_accuracy": result["history"]["best_val_accuracy"],
                "test_accuracy": result["test_metrics"]["accuracy"],
            }
        )
    return pd.DataFrame(records)


def run_complete_variant_report(
    student_models=None,
    experiments=None,
    teacher_pretrained=True,
    student_pretrained=True,
    teacher_training_config=None,
    student_training_config=None,
    baseline_save_dir=None,
    teacher_save_dir=None,
    student_save_dir=None,
    variant_names=None,
    show_augments=True,
    output_dir="outputs/reports/full_variant_report",
):
    output_dir = Path(output_dir)
    baseline_dir = output_dir / "baselines"
    variants_dir = output_dir / "variants"
    tables_dir = output_dir / "tables"
    data_dir = output_dir / "data"
    baseline_data_dir = data_dir / "baselines"
    variant_data_dir = data_dir / "variants"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    baseline_data_dir.mkdir(parents=True, exist_ok=True)
    variant_data_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = run_student_baselines(
        student_models=student_models,
        pretrained=student_pretrained,
        training_config=student_training_config,
        save_dir=baseline_save_dir,
        context=None,
        show_augments=show_augments,
    )

    print("\n=== Student Baselines ===")
    baseline_df = summarize_baseline_results(baseline_results)
    display(baseline_df)
    baseline_df.to_csv(tables_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")
    fig_baseline_bar, _ = plot_result_metric_bars(
        baseline_df["student"].tolist(),
        baseline_df["test_accuracy"].tolist(),
        title="Baseline Acc",
    )
    save_figure(fig_baseline_bar, baseline_dir / "baseline_test_accuracy.png")
    plt.show()
    for result in baseline_results:
        figures = visualize_baseline_result(result, class_names=CLASS_NAMES, normalize_confusion=False)
        model_slug = slugify(result["model_name"])
        save_figure(figures["history_figure"], baseline_dir / f"{model_slug}_training_history.png")
        save_figure(figures["confusion_figure"], baseline_dir / f"{model_slug}_confusion_matrix.png")
        export_history_csv(
            result["history"],
            baseline_data_dir / f"{model_slug}_history.csv",
            metadata={
                "run_type": "baseline",
                "model_name": result["model_name"],
            },
        )

    suite_results = run_all_distillation_variant_suites(
        baseline_results=baseline_results,
        experiments=experiments,
        teacher_pretrained=teacher_pretrained,
        student_pretrained=student_pretrained,
        teacher_training_config=teacher_training_config,
        student_training_config=student_training_config,
        teacher_save_dir=teacher_save_dir,
        student_save_dir=student_save_dir,
        variant_names=variant_names,
        show_augments=False,
    )

    suite_tables = {}
    history_index_records = []
    for suite_result in suite_results:
        key = f"{suite_result['teacher_name']} -> {suite_result['student_name']}"
        key_slug = slugify(key)
        pair_label = f"{short_name(suite_result['teacher_name'])}->{short_name(suite_result['student_name'])}"
        pair_data_dir = variant_data_dir / key_slug
        pair_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {key} ===")
        summary_df = summarize_variant_suite(suite_result, metric_key="test_accuracy")
        display(summary_df)
        suite_tables[key] = summary_df
        summary_df.to_csv(tables_dir / f"{key_slug}_variant_results.csv", index=False, encoding="utf-8-sig")

        baseline_history_path = pair_data_dir / "baseline_history.csv"
        export_history_csv(
            suite_result["baseline_result"]["history"],
            baseline_history_path,
            metadata={
                "run_type": "baseline",
                "teacher_name": suite_result["teacher_name"],
                "student_name": suite_result["student_name"],
                "variant_name": "baseline",
                "variant_label": "Baseline",
            },
        )
        history_index_records.append(
            {
                "teacher_name": suite_result["teacher_name"],
                "student_name": suite_result["student_name"],
                "variant_name": "baseline",
                "variant_label": "Baseline",
                "history_csv": str(baseline_history_path),
            }
        )

        labels = summary_df["variant"].tolist()
        compared_values = summary_df["test_accuracy"].tolist()
        baseline_values = summary_df["baseline_accuracy"].tolist()

        fig_accuracy_bar, _ = plot_result_metric_bars(
            labels,
            compared_values,
            title=f"{pair_label} Base vs Distill Acc",
        )
        save_figure(fig_accuracy_bar, variants_dir / f"{key_slug}_accuracy_bar.png")
        plt.show()
        fig_gain_bar, _ = plot_distillation_gains(
            labels,
            baseline_values,
            compared_values,
            title=f"{pair_label} Distill Gain",
        )
        save_figure(fig_gain_bar, variants_dir / f"{key_slug}_gain_bar.png")
        plt.show()

        fig_curves = compare_training_curves(
            [suite_result["baseline_result"]] + suite_result["variant_results"],
            title=f"{pair_label} Variant Acc",
        )
        save_figure(fig_curves, variants_dir / f"{key_slug}_training_curves.png")
        fig_confusions = compare_confusion_matrices(
            [suite_result["baseline_result"]] + suite_result["variant_results"],
            class_names=CLASS_NAMES,
            normalize=False,
        )
        save_figure(fig_confusions, variants_dir / f"{key_slug}_confusion_matrices.png")

        for variant_result in suite_result["variant_results"]:
            variant_slug = slugify(variant_result["variant_name"])
            history_path = pair_data_dir / f"{variant_slug}_history.csv"
            export_history_csv(
                variant_result["history"],
                history_path,
                metadata={
                    "run_type": "distillation",
                    "teacher_name": suite_result["teacher_name"],
                    "student_name": suite_result["student_name"],
                    "variant_name": variant_result["variant_name"],
                    "variant_label": variant_result["variant_label"],
                    "feature_loss_type": variant_result.get("feature_loss_type", "none"),
                },
            )
            history_index_records.append(
                {
                    "teacher_name": suite_result["teacher_name"],
                    "student_name": suite_result["student_name"],
                    "variant_name": variant_result["variant_name"],
                    "variant_label": variant_result["variant_label"],
                    "history_csv": str(history_path),
                }
            )

    history_index_df = pd.DataFrame(history_index_records)
    history_index_df.to_csv(data_dir / "history_index.csv", index=False, encoding="utf-8-sig")

    return {
        "baseline_results": baseline_results,
        "baseline_table": baseline_df,
        "suite_results": suite_results,
        "suite_tables": suite_tables,
        "history_index": history_index_df,
        "output_dir": output_dir,
    }


def run_single_pair_parameter_sweep(
    teacher_name,
    student_name,
    parameter_name="temperature",
    parameter_values=None,
    distill_mode="logits_only",
    teacher_pretrained=True,
    student_pretrained=True,
    teacher_training_config=None,
    student_training_config=None,
    baseline_save_dir=None,
    teacher_save_dir=None,
    student_save_dir=None,
    show_augments=True,
    output_dir="outputs/reports/single_pair_parameter_sweep",
):
    if parameter_values is None:
        parameter_values = [2.0, 3.0, 4.0, 5.0]

    teacher_name = normalize_model_name(teacher_name)
    student_name = normalize_model_name(student_name)
    pair_label = f"{short_name(teacher_name)}->{short_name(student_name)}"
    output_dir = Path(output_dir) / f"{slugify(teacher_name)}_to_{slugify(student_name)}_{slugify(distill_mode)}_{slugify(parameter_name)}"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    context = prepare_context(show_augments=show_augments)
    baseline_results = run_student_baselines(
        student_models=[student_name],
        pretrained=student_pretrained,
        training_config=student_training_config,
        save_dir=baseline_save_dir,
        context=context,
        show_augments=False,
    )
    baseline_result = baseline_results[0]

    teacher_save_path = None
    if teacher_save_dir is not None:
        teacher_save_path = f"{teacher_save_dir}/{teacher_name}.pt"
    teacher_result = train_teacher_model(
        model_name=teacher_name,
        pretrained=teacher_pretrained,
        save_path=teacher_save_path,
        training_config=teacher_training_config,
        context=context,
        show_augments=False,
    )

    baseline_figures = visualize_baseline_result(
        baseline_result,
        class_names=CLASS_NAMES,
        normalize_confusion=False,
    )
    save_figure(baseline_figures["history_figure"], figures_dir / "baseline_training_history.png")
    save_figure(baseline_figures["confusion_figure"], figures_dir / "baseline_confusion_matrix.png")
    export_history_csv(
        baseline_result["history"],
        tables_dir / "baseline_history.csv",
        metadata={
            "run_type": "baseline",
            "teacher_name": teacher_name,
            "student_name": student_name,
            "variant_name": "baseline",
            "variant_label": "Baseline",
        },
    )

    base_variant_config = build_variant_distillation_config(student_name, distill_mode)
    variant_meta = DISTILLATION_VARIANT_LIBRARY[distill_mode]
    sweep_results = []

    for parameter_value in parameter_values:
        override_config = {
            "student_specific_config": {
                student_name: {
                    parameter_name: parameter_value,
                }
            }
        }
        distillation_config = merge_distillation_configs(base_variant_config, override_config)
        student_save_path = None
        if student_save_dir is not None:
            student_save_path = f"{student_save_dir}/{teacher_name}_to_{student_name}_{distill_mode}_{parameter_name}_{parameter_value}.pt"

        student_result = train_distilled_model(
            teacher_name=teacher_name,
            student_name=student_name,
            teacher_model=teacher_result["model"],
            pretrained_teacher=False,
            pretrained_student=student_pretrained,
            training_config=student_training_config,
            distillation_config=distillation_config,
            save_path=student_save_path,
            use_feature_distill=variant_meta["use_feature_distill"],
            context=context,
            show_augments=False,
        )
        student_result["variant_name"] = distill_mode
        student_result["variant_label"] = f"{variant_meta['label']} | {parameter_name}={parameter_value}"
        student_result["sweep_parameter"] = parameter_name
        student_result["sweep_value"] = parameter_value
        sweep_results.append(student_result)

    records = [
        {
            "teacher": teacher_name,
            "student": student_name,
            "mode": "Baseline",
            "parameter_name": parameter_name,
            "parameter_value": "baseline",
            "best_val_accuracy": baseline_result["history"]["best_val_accuracy"],
            "test_accuracy": baseline_result["test_metrics"]["accuracy"],
            "gain_vs_baseline": 0.0,
        }
    ]
    baseline_accuracy = baseline_result["test_metrics"]["accuracy"]
    for result in sweep_results:
        records.append(
            {
                "teacher": teacher_name,
                "student": student_name,
                "mode": result["variant_label"],
                "parameter_name": parameter_name,
                "parameter_value": result["sweep_value"],
                "best_val_accuracy": result["history"]["best_val_accuracy"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "gain_vs_baseline": result["test_metrics"]["accuracy"] - baseline_accuracy,
            }
        )
    summary_df = pd.DataFrame(records)
    display(summary_df)
    summary_df.to_csv(tables_dir / "sweep_results.csv", index=False, encoding="utf-8-sig")

    labels = summary_df["mode"].tolist()
    accuracy_values = summary_df["test_accuracy"].tolist()
    baseline_values = [baseline_accuracy] * len(labels)

    fig_accuracy_bar, _ = plot_result_metric_bars(
        labels,
        accuracy_values,
        title=f"{pair_label} {distill_mode} {parameter_name} Acc",
    )
    save_figure(fig_accuracy_bar, figures_dir / "sweep_accuracy_bar.png")
    plt.show()

    fig_gain_bar, _ = plot_distillation_gains(
        labels,
        baseline_values,
        accuracy_values,
        title=f"{pair_label} {distill_mode} {parameter_name} Gain",
    )
    save_figure(fig_gain_bar, figures_dir / "sweep_gain_bar.png")
    plt.show()

    comparison_results = [baseline_result] + sweep_results
    fig_curves = compare_training_curves(
        comparison_results,
        title=f"{pair_label} {distill_mode} {parameter_name}",
    )
    save_figure(fig_curves, figures_dir / "sweep_training_curves.png")

    fig_confusions = compare_confusion_matrices(
        comparison_results,
        class_names=CLASS_NAMES,
        normalize=False,
    )
    save_figure(fig_confusions, figures_dir / "sweep_confusion_matrices.png")

    history_index_records = [
        {
            "teacher_name": teacher_name,
            "student_name": student_name,
            "variant_name": "baseline",
            "variant_label": "Baseline",
            "history_csv": str(tables_dir / "baseline_history.csv"),
        }
    ]
    for result in sweep_results:
        history_path = tables_dir / f"{slugify(str(result['sweep_value']))}_history.csv"
        export_history_csv(
            result["history"],
            history_path,
            metadata={
                "run_type": "distillation",
                "teacher_name": teacher_name,
                "student_name": student_name,
                "variant_name": result["variant_name"],
                "variant_label": result["variant_label"],
                "parameter_name": result["sweep_parameter"],
                "parameter_value": result["sweep_value"],
            },
        )
        history_index_records.append(
            {
                "teacher_name": teacher_name,
                "student_name": student_name,
                "variant_name": result["variant_name"],
                "variant_label": result["variant_label"],
                "parameter_name": result["sweep_parameter"],
                "parameter_value": result["sweep_value"],
                "history_csv": str(history_path),
            }
        )
    history_index_df = pd.DataFrame(history_index_records)
    history_index_df.to_csv(tables_dir / "history_index.csv", index=False, encoding="utf-8-sig")

    return {
        "teacher_name": teacher_name,
        "student_name": student_name,
        "distill_mode": distill_mode,
        "parameter_name": parameter_name,
        "parameter_values": parameter_values,
        "baseline_result": baseline_result,
        "teacher_result": teacher_result,
        "sweep_results": sweep_results,
        "summary_table": summary_df,
        "history_index": history_index_df,
        "output_dir": output_dir,
    }


def run_single_pair_multi_sweep(
    teacher_name,
    student_name,
    distill_mode="logits_only",
    sweep_types=None,
    teacher_pretrained=True,
    student_pretrained=True,
    teacher_training_config=None,
    student_training_config=None,
    baseline_save_dir=None,
    teacher_save_dir=None,
    student_save_dir=None,
    show_augments=True,
    output_dir="outputs/reports/single_pair_multi_sweep",
):
    teacher_name = normalize_model_name(teacher_name)
    student_name = normalize_model_name(student_name)

    preset_sweeps = SWEEP_TYPE_LIBRARY.get(distill_mode, {})
    if sweep_types is None:
        sweep_types = list(preset_sweeps.keys())[:4]

    base_output_dir = Path(output_dir) / f"{slugify(teacher_name)}_to_{slugify(student_name)}_{slugify(distill_mode)}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    summary_records = []
    for sweep_type in sweep_types:
        parameter_values = preset_sweeps.get(sweep_type)
        if parameter_values is None:
            raise ValueError(f"Unsupported sweep_type '{sweep_type}' for distill_mode '{distill_mode}'")

        sweep_report = run_single_pair_parameter_sweep(
            teacher_name=teacher_name,
            student_name=student_name,
            parameter_name=sweep_type,
            parameter_values=parameter_values,
            distill_mode=distill_mode,
            teacher_pretrained=teacher_pretrained,
            student_pretrained=student_pretrained,
            teacher_training_config=teacher_training_config,
            student_training_config=student_training_config,
            baseline_save_dir=baseline_save_dir,
            teacher_save_dir=teacher_save_dir,
            student_save_dir=student_save_dir,
            show_augments=show_augments if not reports else False,
            output_dir=base_output_dir,
        )
        reports[sweep_type] = sweep_report

        best_row = sweep_report["summary_table"].sort_values("test_accuracy", ascending=False).iloc[0]
        summary_records.append(
            {
                "teacher": teacher_name,
                "student": student_name,
                "distill_mode": distill_mode,
                "sweep_type": sweep_type,
                "best_mode": best_row["mode"],
                "best_test_accuracy": best_row["test_accuracy"],
                "best_val_accuracy": best_row["best_val_accuracy"],
                "gain_vs_baseline": best_row["gain_vs_baseline"],
            }
        )

    summary_df = pd.DataFrame(summary_records)
    display(summary_df)
    summary_path = base_output_dir / "multi_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return {
        "teacher_name": teacher_name,
        "student_name": student_name,
        "distill_mode": distill_mode,
        "sweep_types": sweep_types,
        "reports": reports,
        "summary_table": summary_df,
        "output_dir": base_output_dir,
    }
