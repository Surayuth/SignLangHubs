import torch
import mlflow
from tqdm import tqdm
from pathlib import Path

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

    return running_loss / len(train_loader)


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

    return running_vloss / len(val_loader)


def train(train_loader, val_loader, model, loss_fn, optimizer, max_epochs, device, log_cfg):
    model.to(device)

    best_epoch = 1
    best_val_loss = float("inf")

    artifact_uri = Path(mlflow.get_artifact_uri())

    for epoch in range(1, max_epochs + 1):
        train_progbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{max_epochs}", ncols=150)

        avg_train_loss = train_one_epoch(
            train_loader, model, loss_fn, optimizer, device, train_progbar
        )

        val_progbar = tqdm(val_loader, desc=f"Val Epoch {epoch}/{max_epochs}", leave=False, ncols=150)
        avg_val_loss = val_one_epoch(val_loader, model, loss_fn, device, val_progbar)

        mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            # log best model
            if log_cfg["best"]:
                torch.save(model.state_dict(), artifact_uri / "best_model.pth")

    # log last model
    if log_cfg["last"]:
        torch.save(model.state_dict(), artifact_uri / "last_model.pth")

    mlflow.log_param("best_epoch", best_epoch)

