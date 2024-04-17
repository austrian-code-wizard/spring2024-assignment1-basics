import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    print(f"L: {logits.shape}")
    print(f"T: {targets.shape}")
    logits -= logits.max(-1, keepdim=True)[0]
    nnl = -logits.gather(-1, targets.unsqueeze(1)).squeeze() + torch.log(
        torch.exp(logits).sum(-1)
    )
    print(f"Gather: {logits.gather(-1, targets.unsqueeze(1)).shape}")
    print(f"NNL: {nnl.shape}")
    return nnl.mean()
