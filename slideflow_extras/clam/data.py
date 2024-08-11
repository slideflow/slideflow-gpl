import torch

from slideflow.mil.data import BagDataset, EncodedDataset, MapDataset

def build_clam_dataset(bags, targets, encoder, bag_size, max_bag_size=None, dtype=torch.float32):
    assert len(bags) == len(targets)

    def _zip(bag, targets):
        features, lengths = bag
        return (features, targets.squeeze(), True), targets.squeeze()

    dataset = MapDataset(
        _zip,
        BagDataset(bags, bag_size=bag_size, max_bag_size=max_bag_size, dtype=dtype),
        EncodedDataset(encoder, targets),
    )
    dataset.encoder = encoder
    return dataset