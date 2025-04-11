import os
import pytest
from transformers import AutoTokenizer
from preprocessing.tokenization import BIOTokenizer
from utils.utils import load_json_data


@pytest.fixture
def data():
	return load_json_data(os.path.join("tests", "test_data", "tokenization.json"))


@pytest.fixture(scope="session")
def tokenizer():
	return AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large", use_fast=True)


@pytest.fixture
def bio_tokenized_concatenated_data(data, tokenizer):
	bio_tokenizer = BIOTokenizer(datasets=[data], tokenizer=tokenizer, max_length=16)
	processed_data = bio_tokenizer.process_files()
	return processed_data


@pytest.fixture
def bio_tokenized_data(data, tokenizer):
	bio_tokenizer = BIOTokenizer(datasets=[data], tokenizer=tokenizer, max_length=16, concatenate_title_abstract=False)
	processed_data = bio_tokenizer.process_files()
	return processed_data


def test():
	print("test")
	assert True, "Test placeholder to ensure pytest runs this file."
