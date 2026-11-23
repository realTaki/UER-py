import torch


def upgrade_gpt_checkpoint(state_dict):
    """Map the old UER GPT embedding and LM head names to modular layers."""
    aliases = {
        "embedding.word_embedding.weight": "embedding.word.embedding.weight",
        "embedding.position_embedding.weight": "embedding.pos.embedding.weight",
        "target.output_layer.weight": "target.lm.output_layer.weight",
        "target.output_layer.bias": "target.lm.output_layer.bias",
    }
    for old_key, new_key in aliases.items():
        if old_key in state_dict:
            state_dict.setdefault(new_key, state_dict.pop(old_key))
    return state_dict


def load_model(model, model_path):
    """Load model weights, including older GPT checkpoints for this branch."""
    state_dict = torch.load(model_path, map_location="cpu")
    unwrapped = model.module if hasattr(model, "module") else model
    if "target.lm.output_layer.weight" in unwrapped.state_dict():
        state_dict = upgrade_gpt_checkpoint(state_dict)
    unwrapped.load_state_dict(state_dict, strict=False)
    return model
