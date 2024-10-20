import yaml
import torch
import mlflow
import argparse
from utils.train import train
from utils.init import init_loader, init_model, init_loss, init_opt, init_scheduler

# Config validation function
def validate_config(config):
    required_train_keys = ["max_epochs"]
    required_tracking_keys = ["exp_name", "run_name", "tags", "uri"]

    for key in required_train_keys:
        if key not in config["train"]:
            raise ValueError(f"Missing key '{key}' in 'train' section.")
    for key in required_tracking_keys:
        if key not in config["tracking"]:
            raise ValueError(f"Missing key '{key}' in 'tracking' section.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="config")
    parser.add_argument("--gpu", action="store_true", help="use gpu")

    args = parser.parse_args()
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load config
    path_config = args.config
    with open(path_config, "r") as f:
        config = yaml.safe_load(f)

    # Validate the config structure
    validate_config(config)

    # Initialize data loaders, model, loss function, and optimizer
    train_loader, val_loader, test_loader, mapper = init_loader(config)
    model = init_model(config, len(mapper))
    loss_fn = init_loss(config)
    optimizer = init_opt(model, config)
    scheduler = init_scheduler(optimizer, config)

    # MLflow experiment setup
    exp_cfg = config["tracking"]
    exp_name = exp_cfg["exp_name"]
    run_name = exp_cfg["run_name"]
    tags = exp_cfg["tags"]
    mlflow.set_tracking_uri(uri=exp_cfg["uri"])

    # Retrieve or create experiment in MLflow
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id

    # Start an MLflow run and log the config file as an artifact
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_artifact(path_config)

        # Training process
        max_epochs = config["train"]["max_epochs"]
        log_cfg = exp_cfg["log_model"]
        train(train_loader, val_loader, model, loss_fn, optimizer, scheduler, max_epochs, device, log_cfg)
