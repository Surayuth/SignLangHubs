from glob import glob
from pathlib import Path
import importlib
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def init_mapper(dataset_src):
    paths = glob(str(dataset_src / "*/*"))
    labels = list(set([Path(p).parent.name for p in paths]))
    mapper = {str(label):i for i, label in enumerate(labels)}
    return mapper

def init_loader(config):
    # create train, val, test dataset
    dset_cfg = config["dataset"]
    dset_params = dset_cfg["params"]
    dataset_src = Path(dset_params["src"])

    mapper = init_mapper(dataset_src)
    transform = getattr(
        importlib.import_module("utils.transform"), dset_cfg["transform"]
    )
    if dset_cfg["split_method"] == "train_val_test":
        paths = glob(str(dataset_src / "*/*"))
        labels = [Path(p).parent.name for p in paths]
        train_val_paths, test_paths = train_test_split(
            paths,
            test_size=dset_params["test_ratio"],
            random_state=dset_params["seed"],
            stratify=labels,
        )
        train_val_labels = [Path(p).parent.name for p in train_val_paths]
        train_paths, val_paths = train_test_split(
            train_val_paths,
            test_size=dset_params["val_ratio"],
            random_state=dset_params["seed"],
            stratify=train_val_labels,
        )

        # display dataset statistic (TODO)
        Dataset = getattr(importlib.import_module("utils.dataset"), dset_cfg["class"])
        train_dataset = Dataset(train_paths, mapper, transform)
        val_dataset = Dataset(val_paths, mapper, transform)
        test_dataset = Dataset(test_paths, mapper, transform)

        loader_cfg = config["train"]["loader_config"]
        train_loader = DataLoader(train_dataset, **loader_cfg)
        val_loader = DataLoader(val_dataset, **loader_cfg)
        test_loader = DataLoader(test_dataset, **loader_cfg)
    else:
        raise NotImplemented

    return train_loader, val_loader, test_loader, mapper


def init_model(config, num_classes):
    imported_module = f"models.{config['train']['model']}"
    model = getattr(
        importlib.import_module(imported_module), "SLModel"
    )(num_classes)
    return model


def init_loss(config):
    loss_cfg = config["train"]["loss_fn"]
    loss_fn = getattr(importlib.import_module(f"utils.loss_fn"), loss_cfg["name"])(
        **loss_cfg["params"]
    )
    return loss_fn


def init_opt(model, config):
    opt_cfg = config["train"]["optimizer"]
    optimizer = getattr(importlib.import_module(f"utils.optimizer"), opt_cfg["name"])(
        model, **opt_cfg["params"]
    )
    return optimizer


def init_scheduler(optimizer, config):
    if "scheduler" in config["train"]:
        scheduler_cfg = config["train"]["scheduler"]
        scheduler = getattr(importlib.import_module(f"utils.scheduler"), scheduler_cfg["name"])(
            optimizer, **scheduler_cfg["params"]
        )
    else:
        scheduler = None
    return scheduler