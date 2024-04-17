import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Logits: B x S x V
    # Targets: B x S

    logits -= logits.max(-1, keepdim=True)[0]
    loss = -logits[torch.arange(logits.shape[0]), targets].squeeze() + torch.log(
        torch.exp(logits).sum(-1)
    )
    return loss.mean()
