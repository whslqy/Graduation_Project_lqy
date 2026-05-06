from copy import deepcopy
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_config import DISTILLATION_CONFIG, TRAINING_CONFIG
from kd_models import (
    build_model,
    forward_with_features,
    get_feature_dim,
    normalize_model_name,
)


class FeatureAdapter(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        hidden_dim = max(student_dim, teacher_dim)
        self.proj = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, features):
        return self.proj(features)


class DistillationWrapper(nn.Module):
    def __init__(self, student_model, adapter=None, feature_loss_type="cosine"):
        super().__init__()
        self.student_model = student_model
        self.adapter = adapter
        self.feature_loss_type = feature_loss_type


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def kd_kl_loss(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def cosine_feature_loss(student_features, teacher_features):
    target = torch.ones(student_features.size(0), device=student_features.device)
    return F.cosine_embedding_loss(student_features, teacher_features, target)


def mse_feature_loss(student_features, teacher_features):
    return F.mse_loss(student_features, teacher_features)


def uses_regressor(feature_loss_type):
    return feature_loss_type in {"mse_regressor", "mse_regressor_cosine"}


def resolve_feature_distill_config(student_name, distillation_config=None):
    kd_config = dict(DISTILLATION_CONFIG)
    if distillation_config is not None:
        kd_config.update(distillation_config)

    student_specific = kd_config.get("student_specific_config", {}).get(student_name, {})
    feature_loss_type = student_specific.get(
        "feature_loss_type",
        kd_config.get("feature_loss_type", "cosine"),
    )
    return kd_config, student_specific, feature_loss_type


def create_distillation_components(student_model, student_name, teacher_model, teacher_name, device, distillation_config=None):
    student_dim = get_feature_dim(student_model, student_name)
    teacher_dim = get_feature_dim(teacher_model, teacher_name)
    _, _, feature_loss_type = resolve_feature_distill_config(student_name, distillation_config)
    adapter = None
    requires_regressor = uses_regressor(feature_loss_type)
    if student_dim != teacher_dim or requires_regressor:
        adapter = FeatureAdapter(student_dim, teacher_dim).to(device)
    wrapper = DistillationWrapper(
        student_model,
        adapter=adapter,
        feature_loss_type=feature_loss_type,
    ).to(device)
    return wrapper


def build_distillation_setup(
    teacher_name,
    student_name,
    num_classes,
    device,
    pretrained_teacher=True,
    pretrained_student=True,
    distillation_config=None,
):
    teacher_name = normalize_model_name(teacher_name)
    student_name = normalize_model_name(student_name)

    teacher_model = build_model(teacher_name, num_classes=num_classes, pretrained=pretrained_teacher).to(device)
    student_model = build_model(student_name, num_classes=num_classes, pretrained=pretrained_student).to(device)
    freeze_model(teacher_model)
    wrapper = create_distillation_components(
        student_model,
        student_name,
        teacher_model,
        teacher_name,
        device,
        distillation_config=distillation_config,
    )
    return teacher_model, wrapper


def build_distillation_wrapper_from_teacher(
    teacher_model,
    teacher_name,
    student_name,
    num_classes,
    device,
    pretrained_student=True,
    distillation_config=None,
):
    teacher_name = normalize_model_name(teacher_name)
    student_name = normalize_model_name(student_name)
    teacher_model = freeze_model(teacher_model.to(device))
    student_model = build_model(student_name, num_classes=num_classes, pretrained=pretrained_student).to(device)
    wrapper = create_distillation_components(
        student_model,
        student_name,
        teacher_model,
        teacher_name,
        device,
        distillation_config=distillation_config,
    )
    return teacher_model, wrapper


def create_distill_optimizer(wrapper, training_config=None):
    config = dict(TRAINING_CONFIG)
    if training_config is not None:
        config.update(training_config)

    parameters = list(wrapper.student_model.parameters())
    if wrapper.adapter is not None:
        parameters += list(wrapper.adapter.parameters())

    optimizer_name = config.get("optimizer", "adam").lower()
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)


def create_distill_scheduler(optimizer, training_config=None):
    config = dict(TRAINING_CONFIG)
    if training_config is not None:
        config.update(training_config)

    scheduler_name = config.get("scheduler", "cosine")
    epochs = config.get("epochs", 1)
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    if scheduler_name == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def merge_student_training_config(student_name, training_config=None, distillation_config=None):
    config = dict(TRAINING_CONFIG)
    if training_config is not None:
        config.update(training_config)

    kd_config = dict(DISTILLATION_CONFIG)
    if distillation_config is not None:
        kd_config.update(distillation_config)

    student_overrides = kd_config.get("student_training_config", {}).get(student_name, {})
    config.update(student_overrides)
    return config


def distillation_forward(wrapper, teacher_model, student_name, teacher_name, images, use_feature_distill=True):
    with torch.no_grad():
        teacher_logits, teacher_features = forward_with_features(teacher_model, teacher_name, images)
    student_logits, student_features = forward_with_features(wrapper.student_model, student_name, images)

    aligned_student_features = student_features
    if wrapper.adapter is not None:
        aligned_student_features = wrapper.adapter(student_features)

    feature_loss = None
    mse_loss = None
    cosine_loss = None
    if use_feature_distill:
        if wrapper.feature_loss_type == "mse_regressor":
            mse_loss = mse_feature_loss(aligned_student_features, teacher_features.detach())
            feature_loss = mse_loss
        elif wrapper.feature_loss_type == "mse_regressor_cosine":
            mse_loss = mse_feature_loss(aligned_student_features, teacher_features.detach())
            cosine_loss = cosine_feature_loss(aligned_student_features, teacher_features.detach())
            feature_loss = mse_loss + cosine_loss
        else:
            cosine_loss = cosine_feature_loss(aligned_student_features, teacher_features.detach())
            feature_loss = cosine_loss

    return {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits.detach(),
        "student_features": aligned_student_features,
        "teacher_features": teacher_features.detach(),
        "feature_loss": feature_loss,
        "mse_loss": mse_loss,
        "cosine_loss": cosine_loss,
    }


def run_distill_epoch(
    wrapper,
    teacher_model,
    teacher_name,
    student_name,
    dataloader,
    device,
    distillation_config=None,
    training_config=None,
    optimizer=None,
    use_feature_distill=True,
    epoch_idx=None,
):
    kd_config = dict(DISTILLATION_CONFIG)
    if distillation_config is not None:
        kd_config.update(distillation_config)

    train_config = dict(TRAINING_CONFIG)
    if training_config is not None:
        train_config.update(training_config)

    is_train = optimizer is not None
    wrapper.train(is_train)
    teacher_model.eval()

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=train_config.get("label_smoothing", 0.0))

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_feature_loss = 0.0
    total_correct = 0
    total_samples = 0

    alpha = kd_config.get("alpha", 0.7)
    temperature = kd_config.get("temperature", 4.0)
    feature_weight = kd_config.get("feature_loss_weight", 0.0) if use_feature_distill else 0.0
    warmup_epochs = kd_config.get("warmup_epochs", 0)
    kd_ramp_epochs = kd_config.get("kd_ramp_epochs", 0)
    feature_start_epoch = kd_config.get("feature_start_epoch", warmup_epochs)

    student_specific = kd_config.get("student_specific_config", {}).get(student_name, {})
    alpha = student_specific.get("alpha", alpha)
    temperature = student_specific.get("temperature", temperature)
    feature_weight = student_specific.get("feature_loss_weight", feature_weight)
    warmup_epochs = student_specific.get("warmup_epochs", warmup_epochs)
    kd_ramp_epochs = student_specific.get("kd_ramp_epochs", kd_ramp_epochs)
    feature_start_epoch = student_specific.get("feature_start_epoch", feature_start_epoch)

    if epoch_idx is not None and epoch_idx < warmup_epochs:
        alpha = 0.0
        feature_weight = 0.0
    elif epoch_idx is not None:
        if kd_ramp_epochs > 0:
            ramp_progress = min(1.0, (epoch_idx - warmup_epochs + 1) / kd_ramp_epochs)
            alpha = alpha * max(ramp_progress, 0.0)
        if epoch_idx < feature_start_epoch:
            feature_weight = 0.0

    gradient_clip_norm = kd_config.get("gradient_clip_norm")

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = distillation_forward(
                wrapper=wrapper,
                teacher_model=teacher_model,
                student_name=student_name,
                teacher_name=teacher_name,
                images=images,
                use_feature_distill=use_feature_distill,
            )

            ce_loss = ce_criterion(outputs["student_logits"], labels)
            kl_loss = kd_kl_loss(outputs["student_logits"], outputs["teacher_logits"], temperature)
            feature_loss = outputs["feature_loss"]
            if feature_loss is None:
                feature_loss = torch.tensor(0.0, device=device)

            total_batch_loss = (1.0 - alpha) * ce_loss + alpha * kl_loss + feature_weight * feature_loss

            if is_train:
                total_batch_loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += total_batch_loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        total_kd_loss += kl_loss.item() * batch_size
        total_feature_loss += feature_loss.item() * batch_size
        total_correct += (outputs["student_logits"].argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        return {
            "loss": 0.0,
            "ce_loss": 0.0,
            "kd_loss": 0.0,
            "feature_loss": 0.0,
            "accuracy": 0.0,
        }

    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce_loss / total_samples,
        "kd_loss": total_kd_loss / total_samples,
        "feature_loss": total_feature_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_distillation(
    teacher_model,
    teacher_name,
    wrapper,
    student_name,
    train_loader,
    val_loader,
    device,
    distillation_config=None,
    training_config=None,
    save_path=None,
    use_feature_distill=True,
    verbose=True,
):
    train_config = merge_student_training_config(
        student_name=student_name,
        training_config=training_config,
        distillation_config=distillation_config,
    )

    wrapper = wrapper.to(device)
    teacher_model = freeze_model(teacher_model.to(device))
    optimizer = create_distill_optimizer(wrapper, train_config)
    scheduler = create_distill_scheduler(optimizer, train_config)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_ce_loss": [],
        "train_kd_loss": [],
        "train_feature_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_ce_loss": [],
        "val_kd_loss": [],
        "val_feature_loss": [],
        "epoch_time_sec": [],
        "best_val_accuracy": 0.0,
    }

    best_state = deepcopy(wrapper.state_dict())
    display_name = f"{teacher_name} -> {student_name}"

    if verbose:
        print(f"Starting distillation: {display_name}")

    for epoch in range(train_config["epochs"]):
        epoch_start = time.time()
        train_metrics = run_distill_epoch(
            wrapper=wrapper,
            teacher_model=teacher_model,
            teacher_name=teacher_name,
            student_name=student_name,
            dataloader=train_loader,
            device=device,
            distillation_config=distillation_config,
            training_config=train_config,
            optimizer=optimizer,
            use_feature_distill=use_feature_distill,
            epoch_idx=epoch,
        )
        val_metrics = run_distill_epoch(
            wrapper=wrapper,
            teacher_model=teacher_model,
            teacher_name=teacher_name,
            student_name=student_name,
            dataloader=val_loader,
            device=device,
            distillation_config=distillation_config,
            training_config=train_config,
            optimizer=None,
            use_feature_distill=use_feature_distill,
            epoch_idx=epoch,
        )
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_ce_loss"].append(train_metrics["ce_loss"])
        history["train_kd_loss"].append(train_metrics["kd_loss"])
        history["train_feature_loss"].append(train_metrics["feature_loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_ce_loss"].append(val_metrics["ce_loss"])
        history["val_kd_loss"].append(val_metrics["kd_loss"])
        history["val_feature_loss"].append(val_metrics["feature_loss"])
        history["epoch_time_sec"].append(epoch_time)

        if val_metrics["accuracy"] >= history["best_val_accuracy"]:
            history["best_val_accuracy"] = val_metrics["accuracy"]
            best_state = deepcopy(wrapper.state_dict())
            if save_path is not None:
                save_distillation_checkpoint(wrapper, optimizer, epoch, history, save_path)

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{train_config['epochs']}] "
                f"distill={display_name} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"train_kd={train_metrics['kd_loss']:.4f} "
                f"train_feat={train_metrics['feature_loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"time={epoch_time:.2f}s"
            )

    wrapper.load_state_dict(best_state)
    if verbose:
        print(
            f"Distillation finished: {display_name}. best_val_accuracy={history['best_val_accuracy']:.4f}"
        )
    return wrapper, history


def evaluate_distilled_student(
    wrapper,
    student_name,
    dataloader,
    device,
):
    wrapper.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = forward_with_features(wrapper.student_model, student_name, images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def save_distillation_checkpoint(wrapper, optimizer, epoch, history, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "student_state_dict": wrapper.student_model.state_dict(),
            "adapter_state_dict": wrapper.adapter.state_dict() if wrapper.adapter is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        save_path,
    )


def load_distillation_checkpoint(wrapper, checkpoint_path, device, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    wrapper.student_model.load_state_dict(checkpoint["student_state_dict"])
    adapter_state_dict = checkpoint.get("adapter_state_dict")
    if wrapper.adapter is not None and adapter_state_dict is not None:
        wrapper.adapter.load_state_dict(adapter_state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
