from copy import deepcopy
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from kd_config import MODEL_TRAINING_CONFIG, TRAINING_CONFIG


def move_batch_to_device(batch, device):
    images, labels = batch
    return images.to(device), labels.to(device)


def create_optimizer(model, training_config=None):
    config = training_config or TRAINING_CONFIG
    optimizer_name = config.get("optimizer", "adam").lower()
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_scheduler(optimizer, training_config=None):
    config = training_config or TRAINING_CONFIG
    scheduler_name = config.get("scheduler", "cosine")
    epochs = config.get("epochs", 1)

    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    if scheduler_name == "none":
        return None
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def create_criterion(training_config=None):
    config = training_config or TRAINING_CONFIG
    return nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))


def merge_model_training_config(model_name=None, training_config=None):
    config = dict(TRAINING_CONFIG)
    if model_name is not None:
        config.update(MODEL_TRAINING_CONFIG.get(model_name, {}))
    if training_config is not None:
        config.update(training_config)
    return config


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        images, labels = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

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


def evaluate_model(model, dataloader, criterion, device):
    return run_epoch(model, dataloader, criterion, device, optimizer=None)


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    model_name=None,
    training_config=None,
    save_path=None,
    verbose=True,
    run_name=None,
):
    config = merge_model_training_config(model_name=model_name, training_config=training_config)

    model = model.to(device)
    criterion = create_criterion(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_sec": [],
        "best_val_accuracy": 0.0,
    }

    best_state = deepcopy(model.state_dict())
    display_name = run_name or model.__class__.__name__

    if verbose:
        print(f"Starting training: {display_name}")

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["epoch_time_sec"].append(epoch_time)

        if val_metrics["accuracy"] >= history["best_val_accuracy"]:
            history["best_val_accuracy"] = val_metrics["accuracy"]
            best_state = deepcopy(model.state_dict())
            if save_path is not None:
                save_checkpoint(model, optimizer, epoch, history, save_path)

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{config['epochs']}] "
                f"model={display_name} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"time={epoch_time:.2f}s"
            )

    model.load_state_dict(best_state)
    if verbose:
        print(
            f"Training finished: {display_name}. best_val_accuracy={history['best_val_accuracy']:.4f}"
        )
    return model, history


def test_classifier(model, test_loader, device, training_config=None):
    criterion = create_criterion(training_config)
    model = model.to(device)
    return evaluate_model(model, test_loader, criterion, device)


def save_checkpoint(model, optimizer, epoch, history, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        save_path,
    )


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
