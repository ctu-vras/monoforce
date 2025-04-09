import torch


def set_device(device: str) -> torch.device:
    """
    Set the device for the torch module
    """
    if "cuda" in device and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif device == "mps" and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
