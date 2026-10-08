"""
Tiny random Llama with the MUSE models' architecture family (LlamaForCausalLM, vocab 32000, untied lm_head) for
CPU tests. Saved in FP32 like the MUSE checkpoints. hidden 256 / intermediate 512 so that INT4 group sizes 32 and
128 divide every Linear input dimension. Run in the paper environment:
    python -m extra.tests.make_tiny --out extra/tests/_out/tiny_target
"""

import argparse

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                      rms_norm_eps=1e-5, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
                      torch_dtype="float32")
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(a.out, safe_serialization=True)
    print(f"saved tiny model ({sum(p.numel() for p in model.parameters())/1e6:.1f} M params) to {a.out}")


if __name__ == "__main__":
    main()
