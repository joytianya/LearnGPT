
from pathlib import Path

import torch
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict





def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def save_model_checkpoint(fabric, model, file_path):
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        fabric.save(file_path, {"model": model})
        fabric.barrier()

        if fabric.global_rank == 0:
            convert_zero_checkpoint_to_fp32_state_dict(file_path, file_path.with_suffix(".path"))

        return
    if isinstance(fabric.strategy, FSDPStrategy):
        save_policy = FullStateDictConfig(
            offload_to_cp = (fabric.world_size > 1 ),
            rank0_only = True
        )
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()
    else:
        state_dict = model.state_dict()
    
    if fabric.global_rank == 0:
        torch.save(state_dict, file_path)
    
    fabric.barrier()