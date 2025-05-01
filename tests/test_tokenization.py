import os
import pytest
from transformers import AutoTokenizer
from preprocessing.ner_tokenizer import BIOTokenizer
from utils.utils import load_json_data, load_bio_labels

MAX_LENGTH = 32
TOKENIZER_MODEL = "michiyasunaga/BioLinkBERT-base"


@pytest.fixture
def data():
	return load_json_data(os.path.join("tests", "test_data", "tokenization.json"))


@pytest.fixture
def id2label():
	_, _, id2label = load_bio_labels()
	return id2label


@pytest.fixture(scope="session")
def tokenizer():
	return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)


@pytest.fixture
def bio_tokenizer_concatenated_data(data, tokenizer):
	bio_tokenizer = BIOTokenizer(datasets=[data], tokenizer=tokenizer, max_length=MAX_LENGTH)
	return bio_tokenizer


@pytest.fixture
def bio_tokenizer(data, tokenizer):
	bio_tokenizer = BIOTokenizer(
		datasets=[data], tokenizer=tokenizer, max_length=MAX_LENGTH, concatenate_title_abstract=False
	)
	return bio_tokenizer


@pytest.fixture
def data_extract_entities_concatenated(data, bio_tokenizer_concatenated_data):
	text, entities = bio_tokenizer_concatenated_data._extract_entities(list(data.values())[0])
	return text, entities


@pytest.fixture
def data_extract_entities(data, bio_tokenizer):
	text, entities = bio_tokenizer._extract_entities(list(data.values())[0])
	return text, entities


def test_title_abstract_concatenation(data_extract_entities_concatenated):
	text, _ = data_extract_entities_concatenated
	assert (
		text[0]
		== "Brain: Dog, cat, and human microbiome - Someweird-endings. Infectious agents   have been long considered to play a role."
	)


def test_entity_indices_after_concatenation(data_extract_entities_concatenated):
	_, entities = data_extract_entities_concatenated
	entities = entities[0]

	assert entities[0]["start_idx"] == 7
	assert entities[0]["end_idx"] == 9

	assert entities[1]["start_idx"] == 27
	assert entities[1]["end_idx"] == 36

	assert entities[2]["start_idx"] == 40
	assert entities[2]["end_idx"] == 56

	assert entities[3]["start_idx"] == 115
	assert entities[3]["end_idx"] == 118

	assert entities[4]["start_idx"] == 70
	assert entities[4]["end_idx"] == 75

	assert entities[5]["start_idx"] == 59
	assert entities[5]["end_idx"] == 68


def test_labels_concatenated(tokenizer, bio_tokenizer_concatenated_data):
	processed_data = bio_tokenizer_concatenated_data.process_files()

	assert len(processed_data) == 1

	assert len(processed_data[0]["labels"]) == MAX_LENGTH

	number_tokens = len([1 for label in processed_data[0]["labels"] if label != -100])
	expected_number_tokens = len(
		[
			1
			for label in tokenizer.convert_ids_to_tokens(processed_data[0]["input_ids"])
			if label not in ["[CLS]", "[SEP]", "[PAD]"]
		]
	)

	assert number_tokens == expected_number_tokens


def test_labels(bio_tokenizer):
	processed_data = bio_tokenizer.process_files()

	assert len(processed_data) == 2


def test_labels_correspond_to_truth(id2label, bio_tokenizer):
	processed_data = bio_tokenizer.process_files()
	labels = []
	seen = set()
	[
		seen.add(label) or labels.append(label)
		for label in processed_data[0]["labels"]
		if label not in [-100, 0] and label not in seen
	]

	assert id2label[labels[0]] == id2label[3]
	assert id2label[labels[1]] == id2label[23]
	assert id2label[labels[2]] == id2label[19]
	assert id2label[labels[3]] == id2label[20]


def test_labels_correspond_to_truth_concatenated(id2label, bio_tokenizer_concatenated_data):
	processed_data = bio_tokenizer_concatenated_data.process_files()
	labels = []
	seen = set()
	[
		seen.add(label) or labels.append(label)
		for label in processed_data[0]["labels"]
		if label not in [-100, 0] and label not in seen
	]

	assert id2label[labels[0]] == id2label[3]
	assert id2label[labels[1]] == id2label[23]
	assert id2label[labels[2]] == id2label[19]
	assert id2label[labels[3]] == id2label[20]
	assert id2label[labels[4]] == id2label[13]
	assert id2label[labels[5]] == id2label[21]
	assert id2label[labels[6]] == id2label[5]
