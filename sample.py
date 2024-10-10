import torch



@torch.no_grad()
# Sample from a starting letter
def sample(model, tokenizer, sequence_start: str, temperature: float, max_length: int, device) -> str:
    
    sequence = [
        torch.tensor(tokenizer.token_to_id(sequence_start), device = device)
    ]
    
    kv_cache = model.get_empty_kv_cache(1, max_length, device)


    for i in range(max_length):
        logits, kv_cache = model.forward(sequence[-1].view(1), kv_cache, i)
        
        outputs = torch.softmax(logits[-1].squeeze() / temperature, dim=0)

        idx = torch.multinomial(outputs, 1)

        sequence.append(idx)
    
    return tokenizer.decode([i.item() for i in sequence])