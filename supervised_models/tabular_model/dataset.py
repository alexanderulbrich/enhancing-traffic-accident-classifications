import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, tabular_data, targets=None): # Make targets optional, default to None
        """
        tabular_data: Tensor of shape (num_samples, tabular_dim)
        targets: Tensor of shape (num_samples,) or None for inference
        """
        self.tabular_data = tabular_data
        self.targets = targets # Store targets, can be None

    def __len__(self):
        return len(self.tabular_data) # Use tabular_data length as it's always provided

    def __getitem__(self, idx):
        # Get the tabular features for this sample.
        tabular_item = self.tabular_data[idx]

        if self.targets is not None: # Check if targets are provided (training case)
            target = self.targets[idx]
            return tabular_item, target # Return target for training
        else: # Inference case: targets is None
            return tabular_item # Don't return target for inference