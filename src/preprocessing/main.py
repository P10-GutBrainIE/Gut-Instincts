import argparse
import os
import yaml
from transformers import AutoTokenizer
from preprocessing.remove_html import remove_html
from preprocessing.data_cleanup import clean_incorrect_text_spans
from preprocessing.tokenization import BIOTokenizer
from utils.utils import load_json_data

parser = argparse.ArgumentParser(description="Clean and preprocess data and apply specified tokenizer")
parser.add_argument("--config", type=str, required=True, help="Name of the tokenizer to use")
args = parser.parse_args()

with open(args.config, "r") as file:
	config = yaml.safe_load(file)

shared_path = os.path.join("data", "Annotations", "Train")
platinum_data = load_json_data(os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"))
gold_data = load_json_data(os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"))
silver_data = load_json_data(os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"))


silver_data = clean_incorrect_text_spans(
	data=silver_data,
	corrections=load_json_data(os.path.join("data", "metadata", "silver_incorrect_annotations.json"))["clean"],
)

platinum_data = remove_html(data=platinum_data)
gold_data = remove_html(data=gold_data)
silver_data = remove_html(data=silver_data)

tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
bio_tokenizer = BIOTokenizer(
	datasets=[platinum_data, gold_data, silver_data], save_path="data_preprocessed", tokenizer=tokenizer
)
bio_tokenizer.process_files()
