"""
Analyze the token position of keyword "text" in the pangu 256-token sequence.

Usage (run with pangu model + tokenizer available):

    python scripts/analyze_pangu_text_token.py \
        --vlm_model_path /path/to/vlm_model \
        --vlm_llm_path /path/to/llm \
        --vlm_vit_path /path/to/vit \
        --prompt "A high-quality photo with clear text" \
        --task_tokens dehalo

Output prints the keyword token index(es) in the 256-length sequence.
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model_path", type=str, required=True)
    parser.add_argument("--vlm_llm_path", type=str, required=True)
    parser.add_argument("--vlm_vit_path", type=str, required=True)
    parser.add_argument("--prompt", type=str,
                        default="A high-quality photo with clear text")
    parser.add_argument("--task_tokens", type=str, default=None)
    parser.add_argument("--keyword", type=str, default="text")
    args = parser.parse_args()

    import torch

    # Adjust this import to your pangu installation
    from models.pangu_vl_1B_v1.inference_und import initialize_model

    # --- Initialize pangu model (returns InterleaveInferencer) ---
    print("[info] initializing pangu model (this may take a while)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferencer = initialize_model(
        model_path=args.vlm_model_path,
        llm_path=args.vlm_llm_path,
        vit_path=args.vlm_vit_path,
        device=device.index if device.type == "cuda" else 0,
    )

    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids

    # --- Task token prefix (mirrors EmbeddingDB logic) ---
    task_prefix = ""
    if args.task_tokens:
        from types import SimpleNamespace
        from framework.instantiate import instantiate
        from omegaconf import OmegaConf

        tv_cfg = OmegaConf.load("configs/model_config/text_encoder/nch_trainable_vector.yaml")
        tv_cfg.params.vlm_text_encoder = SimpleNamespace(tokenizer=None, model=None)
        tv = instantiate(tv_cfg)
        if args.task_tokens in tv.task_tokens:
            task_prefix = tv.task_tokens[args.task_tokens]
            print(f"[info] task_prefix for '{args.task_tokens}': {task_prefix!r}")

    full_prompt = task_prefix + args.prompt
    print(f"[info] full prompt: {full_prompt!r}")

    # --- Raw tokenization (for reference) ---
    raw_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(raw_ids)
    print(f"[info] raw tokenization ({len(raw_tokens)} tokens):")
    keyword_raw_indices = []
    for i, tok in enumerate(raw_tokens):
        marker = "  <== KEYWORD" if args.keyword.lower() in tok.lower() else ""
        print(f"  [{i:3d}] id={raw_ids[i]} {tok!r}{marker}")
        if args.keyword.lower() in tok.lower():
            keyword_raw_indices.append(i)
    print(f"[info] '{args.keyword}' raw token indices: {keyword_raw_indices}")

    # --- Run prepare_prompts to get packed_text_ids ---
    gen_context = {
        'kv_lens': [0],
        'ropes': [0],
        'past_key_values': inferencer.init_gen_context()['past_key_values'],
    }

    generation_input, kv_lens, ropes = inferencer.model.prepare_prompts(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        prompts=[full_prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    print(f"[info] generation_input keys: {list(generation_input.keys())}")

    # --- Strategy 1: inspect packed_text_ids ---
    packed_text_ids = generation_input.get("packed_text_ids")
    if packed_text_ids is not None:
        ids_seq = packed_text_ids.squeeze().tolist()
        print(f"[info] packed_text_ids length: {len(ids_seq)}")
        print(f"[info] first 50 ids: {ids_seq[:50]}")
        print(f"[info] last 10 ids: {ids_seq[-10:]}")

        keyword_token_ids = set()
        for idx in keyword_raw_indices:
            keyword_token_ids.add(raw_ids[idx])
        print(f"[info] keyword token ids: {keyword_token_ids} (from raw indices {keyword_raw_indices})")

        matches = [i for i, tid in enumerate(ids_seq) if tid in keyword_token_ids]
        if matches:
            print(f"\n>>> RESULT: keyword '{args.keyword}' at packed positions: {matches}")
            for m in matches:
                tok = tokenizer.convert_ids_to_tokens(ids_seq[m])
                print(f"    [{m:3d}] id={ids_seq[m]} token={tok!r}")
            print("\n[usage] Set these indices as taca.text_token_indices in the training config.")
        else:
            # Try partial matching
            print(f"[warn] no exact match; trying partial substring match...")
            keyword_str = args.keyword.lower()
            partial_matches = []
            for i, tid in enumerate(ids_seq[:100]):
                try:
                    tok = tokenizer.convert_ids_to_tokens(tid)
                    if keyword_str in tok.lower():
                        partial_matches.append((i, tid, tok))
                except Exception:
                    pass
            if partial_matches:
                print(f"\n>>> RESULT (partial): keyword '{args.keyword}' at positions: {[m[0] for m in partial_matches]}")
                for m in partial_matches:
                    print(f"    [{m[0]:3d}] id={m[1]} token={m[2]!r}")
            else:
                print(f"\n[info] No 'text' token found in packed_text_ids.")
                print(f"    Check raw tokenization for actual keyword token id ({raw_ids[keyword_raw_indices[0]] if keyword_raw_indices else 'N/A'}).")
    else:
        print("[warn] packed_text_ids not found. Use offset estimation.")
        task_len = len(tokenizer.encode(task_prefix, add_special_tokens=False)) if task_prefix else 0
        prompt_len = len(raw_ids)
        print(f"\n[info] raw prompt length: {prompt_len} tokens (incl. task_prefix: {task_len})")
        print(f"    Keyword position in 256 sequence = system_prompt_offset + {keyword_raw_indices}")
        print(f"    Determine offset by inspecting prepare_prompts source.")


if __name__ == "__main__":
    main()
