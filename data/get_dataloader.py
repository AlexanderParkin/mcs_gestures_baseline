import torch
from . import dataset


def get_dataloaders(config):
    print("Preparing train reader...")
    train_dataset = dataset.GestureDataset(config, is_train=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print("Done.")

    print("Preparing valid reader...")
    val_dataset = dataset.GestureDataset(config, is_train=False)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True
    )
    print("Done.")
    return train_loader, valid_loader
