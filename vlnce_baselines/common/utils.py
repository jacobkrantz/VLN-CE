from typing import Dict, List

import torch


def transform_obs(
    observations: List[Dict], instruction_sensor_uuid: str
) -> Dict[str, torch.Tensor]:
    r"""Extracts instruction tokens from an instruction sensor and
    transposes a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        instruction_sensor_uuid: name of the instructoin sensor to
            extract from.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    for i in range(len(observations)):
        observations[i][instruction_sensor_uuid] = observations[i][
            instruction_sensor_uuid
        ]["tokens"]
    return observations
