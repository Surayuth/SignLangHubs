from torch.optim import lr_scheduler


def StepLR(optimizer, **kwargs):
    scheduler = lr_scheduler.StepLR(optimizer, **kwargs)
    return scheduler


def ReduceLROnPlateau(optimizer, **kwargs):
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    return scheduler
