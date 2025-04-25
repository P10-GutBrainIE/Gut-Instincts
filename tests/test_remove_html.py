import os
import pytest
from utils.utils import load_json_data
from preprocessing.remove_html import remove_html_tags


@pytest.fixture
def cleaned_data():
	raw_data = load_json_data(file_path=os.path.join("tests", "test_data", "remove_html.json"))
	return remove_html_tags(data=raw_data)


def test_remove_html_title(cleaned_data):
	assert cleaned_data["paper_id"]["metadata"]["title"] == "This is a nested tag sentence"


def test_remove_html_abstract(cleaned_data):
	assert cleaned_data["paper_id"]["metadata"]["abstract"] == "My name is John2"


def test_remove_html_entities(cleaned_data):
	entities = cleaned_data["paper_id"]["entities"]

	assert entities[0]["text_span"] == "nested tag sentence"
	assert entities[0]["start_idx"] == 10
	assert entities[0]["end_idx"] == 28

	assert entities[1]["text_span"] == "John2"
	assert entities[1]["start_idx"] == 11
	assert entities[1]["end_idx"] == 15


def test_remove_html_relations(cleaned_data):
	relations = cleaned_data["paper_id"]["relations"]

	assert relations[0]["subject_text_span"] == "This"
	assert relations[0]["subject_start_idx"] == 0
	assert relations[0]["subject_end_idx"] == 3

	assert relations[0]["object_text_span"] == "tag"
	assert relations[0]["object_start_idx"] == 17
	assert relations[0]["object_end_idx"] == 19


def test_remove_html_metadata_no_html(cleaned_data):
	metadata_no_html = cleaned_data["paper_id_no_html"]["metadata"]

	assert metadata_no_html["title"] == "This is a plain text title."
	assert metadata_no_html["abstract"] == "This is a plain text abstract."


def test_remove_html_entities_no_html(cleaned_data):
	entities_no_html = cleaned_data["paper_id_no_html"]["entities"]

	assert entities_no_html[0]["text_span"] == "plain"
	assert entities_no_html[0]["start_idx"] == 10
	assert entities_no_html[0]["end_idx"] == 14
