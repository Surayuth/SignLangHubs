from torch.optim import lr_scheduler

def StepLR(optimizer, **kwargs):
    scheduler = lr_scheduler.StepLR(optimizer, **kwargs)
    return scheduler