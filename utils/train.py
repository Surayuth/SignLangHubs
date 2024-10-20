import torch
import mlflow
from tqdm import tqdm
from pathlib import Path


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_one_epoch(train_loader, model, loss_fn, optimizer, device, train_progbar):
    model.train()
    running_loss = 0

    for inputs, labels in train_progbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        train_progbar.set_postfix(loss=loss.item())

    train_metrics = {
        "avg_train_loss": running_loss / len(train_loader)
    }
    return train_metrics


def val_one_epoch(val_loader, model, loss_fn, device, val_progbar):
    model.eval()
    running_vloss = 0

    with torch.no_grad():
        for vinputs, vlabels in val_progbar:
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

            val_progbar.set_postfix(loss=vloss.item())
    
    avg_val_loss = running_vloss / len(val_loader)
    val_metrics = {
        "avg_val_loss": avg_val_loss
    }
    return val_metrics


def train(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    max_epochs,
    device,
    log_cfg,
    eval_metric
):
    model.to(device)

    best_epoch = 1
    best_val_score = float("inf")

    artifact_uri = Path(mlflow.get_artifact_uri())

    for epoch in range(1, max_epochs + 1):
        train_progbar = tqdm(
            train_loader, desc=f"Train Epoch {epoch}/{max_epochs}", ncols=150
        )

        train_metrics = train_one_epoch(
            train_loader, model, loss_fn, optimizer, device, train_progbar
        )

        val_progbar = tqdm(
            val_loader, desc=f"Val Epoch {epoch}/{max_epochs}", leave=False, ncols=150
        )
        val_metrics = val_one_epoch(val_loader, model, loss_fn, device, val_progbar)

        mlflow.log_metric("lr", get_lr(optimizer), step=epoch)
        mlflow.log_metrics(train_metrics, step=epoch)
        mlflow.log_metrics(val_metrics, step=epoch)

        if val_metrics[eval_metric] < best_val_score:
            best_val_score = val_metrics[eval_metric]
            best_epoch = epoch
            # log best model
            if log_cfg["best"]:
                torch.save(model.state_dict(), artifact_uri / "best_model.pth")

        scheduler.step(val_metrics)

    # log last model
    if log_cfg["last"]:
        torch.save(model.state_dict(), artifact_uri / "last_model.pth")

    mlflow.log_param("best_epoch", best_epoch)
