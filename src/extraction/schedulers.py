"""
Schedulers.
"""
from torch import optim

schedulers = {
    "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau
}
