# Slideflow-Extras - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-Extras.
#
# Slideflow-Extras is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Slideflow-Extras is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Slideflow-Extras. If not, see <https://www.gnu.org/licenses/>.

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