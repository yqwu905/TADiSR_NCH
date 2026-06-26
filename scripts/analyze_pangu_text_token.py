"""
Analyze the token position of keyword "text" in the pangu 256-token sequence.

TADiSR's TACA needs to locate the "text" keyword token in the text encoder
output sequence to extract the corresponding attention. With offline
EmbeddingDB, the 256-length encoder_hidden_states are precomputed, so we run
this script ONCE with the real pangu model to find the fixed index, then
hardcode it into the training config.

Usage (run in an environment with pangu model + tokenizer available):

    python scripts/analyze_pangu_text_token.py \
        --vlm_model_path /path/to/vlm_model \
        --vlm_llm_path /path/to/llm \
        --vlm_vit_path /path/to/vit \
        --vocab_file /path/to/spiece.model \
        --prompt "A high-quality photo with clear text" \
        --task_tokens dehalo

Output prints the token index(es) of "text" in the 256-sequence, which should
be filled into configs/tadisr_*.yaml as `taca.text_token_indices`.

NOTE: This script depends on the pangu inference stack
(models.pangu_vl_1B_v1.inference_und.initialize_model) which is NOT part of
this repo. Adjust the import path and initialize_model signature to match
your local pangu installation.
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model_path", type=str, required=True)
    parser.add_argument("--vlm_llm_path", type=str, required=True)
    parser.add_argument("--vlm_vit_path", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="SentencePiece vocab file for PanguTokenizer")
    parser.add_argument("--prompt", type=str,
                        default="A high-quality photo with clear text")
    parser.add_argument("--task_tokens", type=str, default=None,
                        help="task key like 'dehalo' to prepend task token prefix")
    parser.add_argument("--keyword", type=str, default="text",
                        help="keyword to locate in the token sequence")
    args = parser.parse_args()

    # --- Lazy imports: only needed when running this script ---
    import torch

    # Adjust this import to match your local pangu package layout.
    try:
        from models.pangu_vl_1B_v1.inference_und import initialize_model
    except ImportError as e:
        print(
            f"[ERROR] cannot import initialize_model: {e}\n"
            "Adjust the import path in this script to match your pangu layout.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Build tokenizer (PanguTokenizer) ---
    # The full PanguTokenizer class was provided separately; place it in
    # models/tokenizer/pangu_tokenizer.py or adjust the import below.
    try:
        from models.tokenizer.pangu_tokenizer import PanguTokenizer
    except ImportError:
        print(
            "[ERROR] cannot import PanguTokenizer. "
            "Place the tokenizer class in models/tokenizer/pangu_tokenizer.py "
            "with proper imports (PreTrainedTokenizer, spm, VOCAB_FILES_NAMES, etc.).",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = PanguTokenizer(vocab_file=args.vocab_file)

    # --- Build task token prefix (mirror EmbeddingDB/TrainableVector logic) ---
    task_prefix = ""
    if args.task_tokens:
        from types import SimpleNamespace
        from framework.instantiate import instantiate
        from omegaconf import OmegaConf

        tv_cfg_path = "configs/model_config/text_encoder/nch_trainable_vector.yaml"
        tv_cfg = OmegaConf.load(tv_cfg_path)
        tv_cfg.params.vlm_text_encoder = SimpleNamespace(tokenizer=None, model=None)
        tv = instantiate(tv_cfg)
        if args.task_tokens in tv.task_tokens:
            task_prefix = tv.task_tokens[args.task_tokens]
            print(f"[info] task_prefix for '{args.task_tokens}': {task_prefix!r}")

    full_prompt = task_prefix + args.prompt
    print(f"[info] full prompt: {full_prompt!r}")

    # --- Tokenize the raw prompt (without pangu wrapping) for reference ---
    raw_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    raw_tokens = [tokenizer.convert_ids_to_ids(i) for i in raw_ids]
    raw_tokens = tokenizer.convert_ids_to_tokens(raw_ids)
    print(f"[info] raw tokenization ({len(raw_tokens)} tokens):")
    for i, tok in enumerate(raw_tokens):
        marker = "  <== KEYWORD" if args.keyword.lower() in tok.lower() else ""
        print(f"  [{i:3d}] {tok!r}{marker}")

    # --- Initialize pangu model and run prepare_prompts ---
    print("[info] initializing pangu model (this may take a while)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm_text_encoder = initialize_model(
        model_path=args.vlm_model_path,
        llm_path=args.vlm_llm_path,
        vit_path=args.vlm_vit_path,
        device=device.index if device.type == "cuda" else 0,
    )

    # new_token_ids: adjust if your pangu version needs different ids.
    new_token_ids = {
        "boi_token_id": vlm_text_encoder.boi_token_id,
        "eoi_token_id": vlm_text_encoder.eoi_token_id,
        "eos_token_id": vlm_text_encoder.eos_token_id,
    }

    gen_input, kv_lens, ropes = vlm_text_encoder.prepare_prompts(
        curr_kvlens=[0],
        curr_rope=[0],
        prompts=[full_prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    # The 256-length sequence is built by prepare_prompts. We need to recover
    # which positions correspond to the prompt tokens. Two strategies:
    #
    # Strategy 1 (preferred): inspect packed_text_ids if present.
    packed_text_ids = gen_input.get("packed_text_ids")
    if packed_text_ids is not None:
        ids_seq = packed_text_ids.squeeze().tolist()
        print(f"[info] packed_text_ids length: {len(ids_seq)}")
        keyword_token_ids = set()
        for t in raw_tokens:
            if args.keyword.lower() in t.lower():
                tid = tokenizer.convert_tokens_to_ids(t)
                keyword_token_ids.add(tid)
        print(f"[info] keyword token ids: {keyword_token_ids}")
        matches = [i for i, tid in enumerate(ids_seq) if tid in keyword_token_ids]
        print(f"\n>>> RESULT: keyword '{args.keyword}' at positions: {matches}")
        print(f"    (in the {len(ids_seq)}-length packed_text_ids sequence)")
        for m in matches:
            print(f"    [{m:3d}] id={ids_seq[m]} token={tokenizer.convert_ids_to_tokens(ids_seq[m])!r}")
        return matches

    # Strategy 2: run forward and inspect output shape / packed_query.
    print("[warn] packed_text_ids not found in prepare_prompts output, "
          "falling back to forward pass.")
    try:
        from models.pangu_vl_1B_v1.cache import NaiveCache
    except ImportError:
        NaiveCache = None
        print("[warn] NaiveCache import failed; provide it if forward needs it.")

    if NaiveCache is None:
        print("[ERROR] cannot run forward without NaiveCache. "
              "Inspect gen_input keys manually:", list(gen_input.keys()))
        sys.exit(1)

    past_key_values = NaiveCache(vlm_text_encoder.config.llm_config.num_hidden_layers)
    output = vlm_text_encoder.forward_cache_update_text(past_key_values, **gen_input)
    packed_query = output["packed_query_sequence"]
    print(f"[info] packed_query_sequence shape: {packed_query.shape}")
    seq_len = packed_query.shape[1] if packed_query.dim() == 3 else packed_query.shape[0]
    print(f"[info] sequence length: {seq_len}")
    print("\n[NOTE] The 256-length encoder_hidden_states used by EmbeddingDB is the")
    print("       reshaped packed_query_sequence ([B, 256, 1536]). To find the keyword")
    print("       position, compare packed_text_ids (strategy 1) which is the reliable way.")
    print("       If prepare_prompts does not expose token ids, you must inspect the")
    print("       prepare_prompts source to map prompt tokens to sequence positions.")


if __name__ == "__main__":
    main()
