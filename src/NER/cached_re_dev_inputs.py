import os
import pickle
from transformers import AutoTokenizer
from utils.utils import load_json_data

SPECIAL_TOKENS = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}

def tokenize_with_entity_markers(text, subj, obj, tokenizer):
    s_start, s_end = subj["start_idx"], subj["end_idx"] + 1
    o_start, o_end = obj["start_idx"], obj["end_idx"] + 1

    if s_start < o_start:
        spans = [(s_start, s_end, "[E1]", "[/E1]"), (o_start, o_end, "[E2]", "[/E2]")]
    else:
        spans = [(o_start, o_end, "[E2]", "[/E2]"), (s_start, s_end, "[E1]", "[/E1]")]

    marked_text = ""
    last_idx = 0
    for start, end, pre_tag, post_tag in spans:
        marked_text += text[last_idx:start]
        marked_text += f"{pre_tag}{text[start:end]}{post_tag}"
        last_idx = end
    marked_text += text[last_idx:]

    encoding = tokenizer(
        marked_text,
        return_attention_mask=True,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

def build_cached_inputs(dev_json_path, tokenizer_name, output_pkl_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    dev_data = load_json_data(dev_json_path)
    cached = []

    for paper_id, content in dev_data.items():
        print(f"Building cached input for paper id: {paper_id}")
        title = content["metadata"]["title"]
        abstract = content["metadata"]["abstract"]
        text = f"{title} {abstract}"
        offset = len(title) + 1

        entities = content.get("entities", [])
        for i, subj in enumerate(entities):
            for j, obj in enumerate(entities):
                if i == j:
                    continue

                subj_adj = subj.copy()
                obj_adj = obj.copy()

                if subj["location"] == "abstract":
                    subj_adj["start_idx"] += offset
                    subj_adj["end_idx"] += offset
                if obj["location"] == "abstract":
                    obj_adj["start_idx"] += offset
                    obj_adj["end_idx"] += offset

                input_ids, attention_mask = tokenize_with_entity_markers(
                    text, subj_adj, obj_adj, tokenizer
                )

                cached.append({
                    "paper_id": paper_id,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "subj": subj,
                    "obj": obj
                })

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(cached, f)

    print(f"Cached {len(cached)} RE inference samples to: {output_pkl_path}")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--json_path", required=True)
	parser.add_argument("--tokenizer_name", required=True)
	parser.add_argument("--output_path", default=None)
	args = parser.parse_args()

	# Construct model-tag-based output path if not provided
	if args.output_path is None:
		model_tag = args.tokenizer_name.replace("/", "_")
		args.output_path = f"data_preprocessed/dev_cached_inputs_{model_tag}.pkl"

	build_cached_inputs(args.json_path, args.tokenizer_name, args.output_path)

