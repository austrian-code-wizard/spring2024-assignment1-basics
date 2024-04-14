import torch
import math
import numpy as np
import numpy.typing as npt
from typing import Optional, Callable, Tuple
from torch.nn import Parameter, Linear, Embedding, Module, ModuleList


class RMSNorm(Module):
    def __init__(
        self, d_model: int, gain_init: torch.Tensor = None, eps: float = 1e-5
    ) -> None:
        super().__init__()
        if gain_init is None:
            gain_init = torch.ones(d_model)
        self.weight = Parameter(torch.zeros((d_model,)))
        self.eps = eps
        self.d_model = d_model

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: (..., d_model)
        """
        numerator = a * self.weight.view(*[1] * (len(a.shape) - 1), self.d_model)
        denominator = torch.sqrt(
            (1 / self.d_model) * torch.square(a).sum(-1, keepdim=True) + self.eps
        )
        return numerator / denominator


class Gelu(Module):
    def __init__(self) -> None:
        super().__init__()
        self._sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.erf(x / self._sqrt_2))


class FFN(Module):
    def __init__(self, d_model: int, d_ff: int = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False)
        self.gelu = Gelu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gelu(self.w1(x)))


def softmax(x: torch.Tensor, dim: int):
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    exp_val = torch.exp(x - max_val)
    return exp_val / exp_val.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    p_drop: float = 0.0,
) -> torch.Tensor:
    """
    Q: B x ... x S x D_k
    K: B x ... x S x D_k
    V: B x ... x S x D_v
    M: S x S
    """

    # pre_scores: B x ... x S x S
    pre_scores = K @ Q.transpose(-1, -2) / math.sqrt(K.shape[-1])
    if mask is not None:
        pre_scores = pre_scores.masked_fill(mask, -torch.inf)
    # scores: B x ... x S x S
    scores = softmax(pre_scores, -1)
    if p_drop > 0:
        scores = torch.nn.functional.dropout(scores, p_drop)
    return scores @ V


class CausalMultiheadSelfAttention(Module):
    def __init__(
        self, d_model: int, num_heads: int, attn_pdrop: float | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        d_k = d_model // num_heads
        self.d_k = d_k

        self.q_proj = Linear(d_k * num_heads, d_model, bias=False)
        self.k_proj = Linear(d_k * num_heads, d_model, bias=False)
        self.v_proj = Linear(d_k * num_heads, d_model, bias=False)

        self.output_proj = Linear(d_k * num_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x ... x S x d_model
        """
        if x.dim() == 2:
            x.unsqueeze(0)

        B, S, _ = x.shape

        queries = x @ self.q_proj.weight.T
        keys = x @ self.k_proj.weight.T
        values = x @ self.v_proj.weight.T

        # q/k/v: B x d_k * num_heads x S

        # quries/keys/values: B x num_heads x S x d_k
        queries = queries.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        keys = keys.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        values = values.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # mask: S x S
        mask = torch.triu(torch.ones((S, S)).bool(), diagonal=1)
        attn = scaled_dot_product_attention(
            queries, keys, values, mask=mask, p_drop=self.attn_pdrop
        )
        # attn: B x h x S x d_k
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        out = self.output_proj(attn)
        return out


class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.dropout = torch.nn.Dropout(residual_pdrop)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x S x d_model
        """
        y = x + self.dropout(self.attn(self.ln1(x)))
        return y + self.dropout(self.ffn(self.ln2(y)))


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(context_length, d_model)
        self.layers = ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(residual_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(
            self.token_embeddings(x)
            + self.position_embeddings(torch.arange(x.shape[1])).unsqueeze(0)
        )
        for block in self.layers:
            x = block(x)
        return self.lm_head(self.ln_final(x))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits -= logits.max(-1, keepdim=True)[0]
    nnl = -logits.gather(1, targets.unsqueeze(1)).squeeze() + torch.log(
        torch.exp(logits).sum(-1)
    )
    return nnl.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 10e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 1
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = state.get(
                    "m", torch.zeros_like(grad)
                )
                v = state.get(
                    "v", torch.zeros_like(grad)
                )
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)
                alpha = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data -= alpha * m / (torch.sqrt(v) + eps)  # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def gradient_clipping(params: list[torch.nn.Parameter], max_l2_norm: float, eps: float = 10e-6):
    for p in params:
        l2 = torch.linalg.norm(p.data)
        if l2 >= max_l2_norm:
            p.data *= max_l2_norm / (l2 + eps)


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(dataset)
    start_indices = np.random.randint(0, N - context_length, size=(batch_size,))
    values = []
    targets = []
    for idx in start_indices:
        values.append(dataset[idx:idx+context_length].tolist())
        targets.append(dataset[idx+1:idx+context_length+1].tolist())
    return torch.tensor(values, device=device), torch.tensor(targets, device=device)