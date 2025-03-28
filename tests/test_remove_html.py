import os
from preprocessing.remove_html import remove_html


def test_remove_html():
	data = remove_html(
		file_paths_dict=[{"test_data": os.path.join("tests", "test_data", "test_data.json")}], save_data=False
	)

	metadata = data["test_data"]["paper_id"]["metadata"]
	entities = data["test_data"]["paper_id"]["entities"]

	assert metadata["title"] == "This is a nested tag sentence"
	assert entities[0]["text_span"] == "nested tag sentence"
	assert entities[0]["start_idx"] == 10
	assert entities[0]["end_idx"] == 28

	assert metadata["abstract"] == "My name is John2"
	assert entities[1]["text_span"] == "John2"
	assert entities[1]["start_idx"] == 11
	assert entities[1]["end_idx"] == 15

	metadata_no_html = data["test_data"]["paper_id_no_html"]["metadata"]
	entities_no_html = data["test_data"]["paper_id_no_html"]["entities"]

	assert metadata_no_html["title"] == "This is a plain text title."
	assert metadata_no_html["abstract"] == "This is a plain text abstract."
	assert entities_no_html[0]["text_span"] == "plain"
	assert entities_no_html[0]["start_idx"] == 10
	assert entities_no_html[0]["end_idx"] == 14
