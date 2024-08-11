import numpy as np

from typing import Union, List, Optional, Tuple, Any, TYPE_CHECKING
from slideflow.mil import utils as mil_utils

if TYPE_CHECKING:
    import torch


def run_inference(
    model: "torch.nn.Module",
    bags: Union[np.ndarray, List[str]],
    attention: bool = False,
    device: Optional[Any] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate CLAM predictions for a list of bags."""

    import torch
    from .model import CLAM_MB, CLAM_SB

    if isinstance(model, (CLAM_MB, CLAM_SB)):
        clam_kw = dict(return_attention=True, return_instance_loss=False)
    else:
        clam_kw = {}
        attention = False

    y_pred = []
    y_att  = []
    device = mil_utils._detect_device(model, device, verbose=True)
    for bag in bags:
        if mil_utils._is_list_of_paths(bag):
            # If bags are passed as a list of paths, load them individually.
            loaded = torch.cat([mil_utils._load_bag(b).to(device) for b in bag], dim=0)
        else:
            loaded = mil_utils._load_bag(bag).to(device)
        with torch.inference_mode():
            logits, att = model(loaded, **clam_kw)
            if attention:
                y_att.append(np.squeeze(att.cpu().numpy()))
            y_pred.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att
