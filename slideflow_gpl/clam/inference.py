# Slideflow-GPL - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-GPL.
#
# Slideflow-GPL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Slideflow-GPL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Slideflow-GPL. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from typing import Union, List, Optional, Tuple, Any, TYPE_CHECKING
from slideflow.mil import utils as mil_utils

if TYPE_CHECKING:
    import torch


def _validate_bags_arg(bags) -> None:
    """Validate that ``bags`` is an iterable of bags, not a single bag.

    :func:`run_inference` iterates ``for bag in bags``, treating each element as
    one slide's bag of shape ``(n_tiles, n_features)``. A bare 2-D tensor/array
    (i.e. a single bag) would instead be iterated tile-by-tile, feeding 1-D
    vectors into the model and failing deep in the forward pass with an opaque
    ``IndexError``. Catch that misuse here with an actionable message.
    """
    import torch

    # A Python list (of paths/tensors/arrays), or an ndarray of bag-path
    # strings, is always a valid collection of bags — iterating yields one bag
    # per element.
    if isinstance(bags, list) or mil_utils._is_list_of_paths(bags):
        return

    # A *numeric* tensor/array is a collection of bags only when it carries an
    # outer bag dimension: shape (n_bags, n_tiles, n_features). A bare 2-D
    # (single bag) or 1-D array, iterated, yields per-tile vectors, so reject it.
    is_tensor = isinstance(bags, torch.Tensor)
    is_numeric_array = (
        isinstance(bags, np.ndarray) and np.issubdtype(bags.dtype, np.number)
    )
    if (is_tensor or is_numeric_array) and bags.ndim < 3:
        raise ValueError(
            "`bags` must be an iterable of bags — a list of bag tensors/paths, "
            "or a 3-D tensor of shape (n_bags, n_tiles, n_features). Got a "
            f"{bags.ndim}-D {type(bags).__name__} of shape {tuple(bags.shape)}, "
            "which looks like a single bag. If you have one bag, wrap it in a "
            "list: predict(model, [bag])."
        )


def run_inference(
    model: "torch.nn.Module",
    bags: Union[np.ndarray, List[str]],
    attention: bool = False,
    device: Optional[Any] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate CLAM predictions for a list of bags."""

    import torch
    from .model import CLAM_MB, CLAM_SB

    _validate_bags_arg(bags)

    if isinstance(model, (CLAM_MB, CLAM_SB)):
        clam_kw = dict(return_attention=True, return_instance_loss=False)
    else:
        clam_kw = {}
        attention = False

    y_pred = []
    y_att  = []
    device = mil_utils._detect_device(model, device, verbose=True)
    with torch.inference_mode():
        for bag in bags:
            loaded = mil_utils._load_bag(bag).to(device)
            logits, att = model(loaded, **clam_kw)
            if attention:
                y_att.append(np.squeeze(att.cpu().numpy()))
            y_pred.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att
