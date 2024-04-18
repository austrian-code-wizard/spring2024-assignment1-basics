import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits -= logits.max(-1, keepdim=True)[0]
    loss = -logits.gather(-1, targets.unsqueeze(-1)).squeeze() + torch.log(
        torch.exp(logits).sum(-1)
    )
    return loss.mean()


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, targets))