import random

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from kd_config import (
    AUG_TYPES,
    BASELINE_MODEL_NAMES,
    CLASS_NAMES,
    DISTILLATION_CONFIG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    TRAINING_CONFIG,
)
from kd_data import all_augmentations, build_dataloaders, describe_samples, load_ckplus_samples
from kd_models import (
    build_model,
    build_teacher_student_models,
    forward_with_features,
    normalize_model_name,
    summarize_pretraining_setup,
)
from distill import (
    build_distillation_setup,
    build_distillation_wrapper_from_teacher,
    evaluate_distilled_student,
    load_distillation_checkpoint,
    train_distillation,
)
from trainer import load_checkpoint, test_classifier, train_classifier
from visualize import collect_predictions, compute_confusion_matrix


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow_batch(inp, title=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis("off")


def preview_augmentations(samples):
    img_path, _ = samples[0]
    pil_img = Image.open(img_path).convert("RGB")
    aug_imgs = all_augmentations(pil_img)
    rows = int(np.ceil(len(aug_imgs) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = np.atleast_2d(axes)
    for idx, (name, aug) in enumerate(aug_imgs):
        row, col = divmod(idx, 4)
        axes[row, col].imshow(aug)
        axes[row, col].set_title(name)
        axes[row, col].axis("off")
    for idx in range(len(aug_imgs), rows * 4):
        row, col = divmod(idx, 4)
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.show()


def prepare_context(show_augments=False):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    samples = load_ckplus_samples()
    print("sample size:", len(samples))
    print("categorical distribution:", describe_samples(samples))

    loaders = build_dataloaders(samples, device, train_aug_types=AUG_TYPES)
    print(f"Use enhanced type collections: {AUG_TYPES}")
    print(
        "Train / Val / Test =",
        len(loaders["train_dataset"]),
        len(loaders["val_dataset"]),
        len(loaders["test_dataset"]),
    )

    if show_augments and samples:
        preview_augmentations(samples)

    print(
        "dataloaders ready:",
        len(loaders["train_dataset"]),
        len(loaders["val_dataset"]),
        len(loaders["test_dataset"]),
    )
    return {
        "samples": samples,
        "loaders": loaders,
        "device": device,
    }


def summarize_model_setup():
    try:
        teacher_models, student_models = build_teacher_student_models(
            DISTILLATION_CONFIG,
            num_classes=len(CLASS_NAMES),
        )
        print("pretraining summary:")
        for line in summarize_pretraining_setup(teacher_models, student_models):
            print("  ", line)
        return {
            "teacher_models": teacher_models,
            "student_models": student_models,
        }
    except ImportError as exc:
        print("pretraining setup check skipped:", exc)
        return {
            "teacher_models": {},
            "student_models": {},
        }


def main(show_augments=False, show_model_summary=True):
    context = prepare_context(show_augments=show_augments)
    if show_model_summary:
        context.update(summarize_model_setup())
    return context


def prepare_baseline_experiment(context=None):
    context = prepare_context(show_augments=False) if context is None else context
    context["available_models"] = BASELINE_MODEL_NAMES
    context["training_config"] = TRAINING_CONFIG
    return context


def train_baseline_model(model_name, pretrained=True, save_path=None, training_config=None, context=None, show_augments=True):
    if context is None:
        context = prepare_context(show_augments=show_augments)
    context = prepare_baseline_experiment(context=context)
    model = build_model(
        model_name=model_name,
        num_classes=len(CLASS_NAMES),
        pretrained=pretrained,
    )
    trained_model, history = train_classifier(
        model=model,
        train_loader=context["loaders"]["train_loader"],
        val_loader=context["loaders"]["val_loader"],
        device=context["device"],
        model_name=model_name,
        training_config=training_config,
        save_path=save_path,
        verbose=True,
        run_name=model_name,
    )
    test_metrics = test_classifier(
        model=trained_model,
        test_loader=context["loaders"]["test_loader"],
        device=context["device"],
        training_config=training_config,
    )
    labels, predictions = collect_predictions(
        trained_model,
        context["loaders"]["test_loader"],
        context["device"],
    )
    confusion_matrix = compute_confusion_matrix(labels, predictions, num_classes=len(CLASS_NAMES))
    print(f"Test accuracy for {model_name}: {test_metrics['accuracy']:.4f}")
    return {
        "model_name": model_name,
        "model": trained_model,
        "history": history,
        "test_metrics": test_metrics,
        "test_labels": labels,
        "test_predictions": predictions,
        "confusion_matrix": confusion_matrix,
        "context": context,
    }


def train_teacher_model(model_name, pretrained=True, save_path=None, training_config=None, context=None, show_augments=True):
    result = train_baseline_model(
        model_name=model_name,
        pretrained=pretrained,
        save_path=save_path,
        training_config=training_config,
        context=context,
        show_augments=show_augments,
    )
    result["role"] = "teacher"
    return result


def run_student_baselines(
    student_models=None,
    pretrained=True,
    training_config=None,
    save_dir=None,
    context=None,
    show_augments=True,
):
    if student_models is None:
        student_models = DISTILLATION_CONFIG["student_models"]

    if context is None:
        context = prepare_context(show_augments=show_augments)
    context = prepare_baseline_experiment(context=context)
    results = []
    for model_name in student_models:
        save_path = None
        if save_dir is not None:
            save_path = f"{save_dir}/{normalize_model_name(model_name)}.pt"
        result = train_baseline_model(
            model_name=model_name,
            pretrained=pretrained,
            save_path=save_path,
            training_config=training_config,
            context=context,
        )
        results.append(result)
    return results


def prepare_distillation_experiment(context=None):
    context = prepare_context(show_augments=False) if context is None else context
    context["distillation_experiments"] = DISTILLATION_CONFIG["experiments"]
    context["training_config"] = TRAINING_CONFIG
    context["distillation_config"] = DISTILLATION_CONFIG
    return context


def train_distilled_model(
    teacher_name,
    student_name,
    teacher_model=None,
    pretrained_teacher=True,
    pretrained_student=True,
    training_config=None,
    distillation_config=None,
    save_path=None,
    use_feature_distill=True,
    context=None,
    show_augments=True,
):
    if context is None:
        context = prepare_context(show_augments=show_augments)
    context = prepare_distillation_experiment(context=context)
    teacher_name = normalize_model_name(teacher_name)
    student_name = normalize_model_name(student_name)

    if teacher_model is None:
        teacher_model, wrapper = build_distillation_setup(
            teacher_name=teacher_name,
            student_name=student_name,
            num_classes=len(CLASS_NAMES),
            device=context["device"],
            pretrained_teacher=pretrained_teacher,
            pretrained_student=pretrained_student,
            distillation_config=distillation_config,
        )
    else:
        teacher_model, wrapper = build_distillation_wrapper_from_teacher(
            teacher_model=teacher_model,
            teacher_name=teacher_name,
            student_name=student_name,
            num_classes=len(CLASS_NAMES),
            device=context["device"],
            pretrained_student=pretrained_student,
            distillation_config=distillation_config,
        )

    trained_wrapper, history = train_distillation(
        teacher_model=teacher_model,
        teacher_name=teacher_name,
        wrapper=wrapper,
        student_name=student_name,
        train_loader=context["loaders"]["train_loader"],
        val_loader=context["loaders"]["val_loader"],
        device=context["device"],
        distillation_config=distillation_config,
        training_config=training_config,
        save_path=save_path,
        use_feature_distill=use_feature_distill,
        verbose=True,
    )
    test_metrics = evaluate_distilled_student(
        wrapper=trained_wrapper,
        student_name=student_name,
        dataloader=context["loaders"]["test_loader"],
        device=context["device"],
    )
    labels, predictions = collect_predictions(
        trained_wrapper.student_model,
        context["loaders"]["test_loader"],
        context["device"],
        forward_fn=lambda model, images: forward_with_features(model, student_name, images)[0],
    )
    confusion_matrix = compute_confusion_matrix(labels, predictions, num_classes=len(CLASS_NAMES))
    print(f"Test accuracy for {teacher_name} -> {student_name}: {test_metrics['accuracy']:.4f}")
    return {
        "teacher_name": teacher_name,
        "student_name": student_name,
        "wrapper": trained_wrapper,
        "student_model": trained_wrapper.student_model,
        "feature_loss_type": trained_wrapper.feature_loss_type,
        "use_feature_distill": use_feature_distill,
        "history": history,
        "test_metrics": test_metrics,
        "test_labels": labels,
        "test_predictions": predictions,
        "confusion_matrix": confusion_matrix,
        "context": context,
    }


def train_teacher_then_student(
    teacher_name,
    student_name,
    teacher_pretrained=True,
    student_pretrained=True,
    teacher_save_path=None,
    student_save_path=None,
    teacher_training_config=None,
    student_training_config=None,
    distillation_config=None,
    use_feature_distill=True,
    show_augments=True,
):
    shared_context = prepare_context(show_augments=show_augments)
    teacher_result = train_teacher_model(
        model_name=teacher_name,
        pretrained=teacher_pretrained,
        save_path=teacher_save_path,
        training_config=teacher_training_config,
        context=shared_context,
    )
    student_result = train_distilled_model(
        teacher_name=teacher_name,
        student_name=student_name,
        teacher_model=teacher_result["model"],
        pretrained_teacher=False,
        pretrained_student=student_pretrained,
        training_config=student_training_config,
        distillation_config=distillation_config,
        save_path=student_save_path,
        use_feature_distill=use_feature_distill,
        context=teacher_result["context"],
    )
    return {
        "teacher": teacher_result,
        "student": student_result,
    }


def run_distillation_experiments(
    experiments=None,
    teacher_pretrained=True,
    student_pretrained=True,
    teacher_training_config=None,
    student_training_config=None,
    distillation_config=None,
    use_feature_distill=True,
    teacher_save_dir=None,
    student_save_dir=None,
    show_augments=True,
):
    if experiments is None:
        experiments = DISTILLATION_CONFIG["experiments"]

    results = []
    trained_teachers = {}
    shared_context = prepare_context(show_augments=show_augments)

    for experiment in experiments:
        teacher_name = normalize_model_name(experiment["teacher"])
        student_name = normalize_model_name(experiment["student"])

        if teacher_name not in trained_teachers:
            teacher_save_path = None
            if teacher_save_dir is not None:
                teacher_save_path = f"{teacher_save_dir}/{teacher_name}.pt"

            teacher_result = train_teacher_model(
                model_name=teacher_name,
                pretrained=teacher_pretrained,
                save_path=teacher_save_path,
                training_config=teacher_training_config,
                context=shared_context,
            )
            trained_teachers[teacher_name] = teacher_result

        student_save_path = None
        if student_save_dir is not None:
            student_save_path = f"{student_save_dir}/{teacher_name}_to_{student_name}.pt"

        student_result = train_distilled_model(
            teacher_name=teacher_name,
            student_name=student_name,
            teacher_model=trained_teachers[teacher_name]["model"],
            pretrained_teacher=False,
            pretrained_student=student_pretrained,
            training_config=student_training_config,
            distillation_config=distillation_config,
            save_path=student_save_path,
            use_feature_distill=use_feature_distill,
            context=shared_context,
        )
        results.append(
            {
                "teacher_name": teacher_name,
                "student_name": student_name,
                "teacher_result": trained_teachers[teacher_name],
                "student_result": student_result,
            }
        )

    return results


def summarize_results(results, pd_module):
    records = []
    for result in results:
        if "teacher_result" in result and "student_result" in result:
            student_result = result["student_result"]
            records.append(
                {
                    "type": "distillation",
                    "teacher": result["teacher_name"],
                    "student": result["student_name"],
                    "best_val_accuracy": student_result["history"]["best_val_accuracy"],
                    "test_accuracy": student_result["test_metrics"]["accuracy"],
                }
            )
        elif "teacher_name" in result and "student_name" in result:
            records.append(
                {
                    "type": "distillation",
                    "teacher": result["teacher_name"],
                    "student": result["student_name"],
                    "best_val_accuracy": result["history"]["best_val_accuracy"],
                    "test_accuracy": result["test_metrics"]["accuracy"],
                }
            )
        else:
            records.append(
                {
                    "type": "baseline",
                    "model": result["model_name"],
                    "best_val_accuracy": result["history"]["best_val_accuracy"],
                    "test_accuracy": result["test_metrics"]["accuracy"],
                }
            )
    return pd_module.DataFrame(records)


def load_baseline_model_checkpoint(model_name, checkpoint_path, device=None, pretrained=False):
    device = get_device() if device is None else device
    model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=pretrained)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    return {
        "model_name": model_name,
        "model": model,
        "checkpoint": checkpoint,
        "device": device,
    }


def load_distilled_model_checkpoint(teacher_name, student_name, checkpoint_path, device=None, pretrained_student=False):
    device = get_device() if device is None else device
    teacher_model, wrapper = build_distillation_setup(
        teacher_name=teacher_name,
        student_name=student_name,
        num_classes=len(CLASS_NAMES),
        device=device,
        pretrained_teacher=False,
        pretrained_student=pretrained_student,
    )
    checkpoint = load_distillation_checkpoint(wrapper, checkpoint_path, device)
    wrapper = wrapper.to(device)
    return {
        "teacher_name": teacher_name,
        "student_name": student_name,
        "teacher_model": teacher_model,
        "wrapper": wrapper,
        "student_model": wrapper.student_model,
        "checkpoint": checkpoint,
        "device": device,
    }
