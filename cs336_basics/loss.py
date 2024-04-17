import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Logits: B x S x V
    # Targets: B x S

    logits -= logits.max(-1, keepdim=True)[0]
    print(f"Logits: {logits.shape}")
    print(f"Targets: {targets.shape}")
    print(f"Max targets: {targets.max()}")
    raise ValueError("test")
    loss = -logits.gather(1, targets.unsqueeze(1)).squeeze() + torch.log(
        torch.exp(logits).sum(-1)
    )
    return loss.mean(0)
