import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM, softmax


def sample(model: TransformerLM, tokenizer: Tokenizer, text: str, max_tokens: int, temperature: float = None, top_p: float = None, eos_token: str = "<|endoftext|>") -> str:
    inputs = tokenizer.encode(text)
    inputs = torch.tensor(inputs).long()
    assert len(inputs.shape) == 1, "inputs must be a 1D tensor"
    inputs = inputs.unsqueeze(0).to(model.device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(inputs)
            logits = logits[0, -1, :]
            if temperature is not None:
                logits /= temperature
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(softmax(sorted_logits), dim=-1)
                cutoff_index = None
                for i, prob in enumerate(cumulative_probs):
                    if prob > top_p:
                        cutoff_index = i + 1
                        break
                assert cutoff_index is not None, "top_p is too high"
                sorted_indices = sorted_indices[:cutoff_index]
                sorted_logits = sorted_logits[:cutoff_index]
                probs = softmax(sorted_logits)
                sorted_idx = torch.multinomial(probs, num_samples=1)
                next_token = sorted_indices[sorted_idx]
            else:
                probs = softmax(logits)
                next_token = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer._inv_vocab[eos_token]:
                break
    return tokenizer.decode(inputs[0].tolist())
