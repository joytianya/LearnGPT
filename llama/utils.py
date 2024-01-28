

from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy






def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def save_model_checkpoint(fabric, model, file_path):
    file_path