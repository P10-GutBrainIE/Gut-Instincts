import streamlit as st
import json


def load_json(uploaded_file):
	"""
	Load JSON data from an uploaded file.
	"""
	return json.load(uploaded_file)


def highlight_text(text, entities, color_map):
	"""
	Highlight entities in the text using HTML span elements with background colors.
	Handles escaping, sorting, and inclusive end_idx logic.
	"""
	entities = sorted(entities, key=lambda x: x["start_idx"])

	highlighted_text = ""
	last_idx = 0

	for entity in entities:
		start_idx = entity["start_idx"]
		end_idx = entity["end_idx"] + 1
		color = color_map.get(entity["label"], "#4F4F4F")

		highlighted_text += text[last_idx:start_idx]

		highlighted_text += f"<span style='background-color: {color}; border-radius: 3px; padding: 2px;'>{text[start_idx:end_idx]}</span>"

		last_idx = end_idx

	highlighted_text += text[last_idx:]

	return highlighted_text


def calculate_differences(true_entities, predicted_entities):
	"""
	Calculate the differences between true entities and predicted entities.
	"""
	differences = []
	true_set = {(e["start_idx"], e["end_idx"], e["label"]) for e in true_entities}
	predicted_set = {(e["start_idx"], e["end_idx"], e["label"]) for e in predicted_entities}

	for start_idx, end_idx, label in true_set - predicted_set:
		differences.append({"start_idx": start_idx, "end_idx": end_idx, "label": "True"})

	for start_idx, end_idx, label in predicted_set - true_set:
		differences.append({"start_idx": start_idx, "end_idx": end_idx, "label": "Predicted"})

	return differences


st.title("NER Prediction Visualizer with Differences")
st.sidebar.title("Options")

uploaded_predictions_file = st.sidebar.file_uploader("Upload Predictions JSON File", type=["json"])

uploaded_dataset_file = st.sidebar.file_uploader("Upload Dataset JSON File (with True Entities)", type=["json"])

if uploaded_predictions_file and uploaded_dataset_file:
	predictions = load_json(uploaded_predictions_file)
	dataset = load_json(uploaded_dataset_file)

	document_ids = list(dataset.keys())

	selected_document = st.sidebar.selectbox("Select Document ID", document_ids)

	# Define a unique color map for each entity type
	color_map = {
		"anatomical location": "#FF4500",  # Orange Red
		"animal": "#FF8C00",  # Dark Orange
		"biomedical technique": "#FFD700",  # Gold
		"bacteria": "#9ACD32",  # Yellow Green
		"chemical": "#32CD32",  # Lime Green
		"dietary supplement": "#00FA9A",  # Medium Spring Green
		"DDF": "#8B0000",  # Dark Red
		"drug": "#1E90FF",  # Dodger Blue
		"food": "#4169E1",  # Royal Blue
		"gene": "#6A5ACD",  # Slate Blue
		"human": "#104E8B",  # Dark Blue
		"microbiome": "#006400",  # Dark Green
		"statistical technique": "#8B4513",  # Saddle Brown
		"True": "#2E8B57",  # Sea Green for differences (true)
		"Predicted": "#800080",  # Purple for differences (predicted)
	}

	# Display document metadata
	document_data = dataset[selected_document]["metadata"]
	st.subheader(f"Document: {selected_document}")
	st.markdown(f"**Title**: {document_data.get('title', 'N/A')}")
	st.markdown(f"**Author(s)**: {document_data.get('author', 'N/A')}")
	st.markdown(f"**Journal**: {document_data.get('journal', 'N/A')}")
	st.markdown(f"**Year**: {document_data.get('year', 'N/A')}")
	st.markdown(f"**Annotator**: {document_data.get('annotator', 'N/A')}")

	# Explanation of colors
	st.markdown("### Color Explanation")
	for entity_type, color in color_map.items():
		st.markdown(
			f"<span style='color: {color}; font-weight: bold;'>{entity_type}</span>",
			unsafe_allow_html=True,
		)

	# Highlight and display the title with true entities
	if "title" in document_data:
		st.markdown("### Highlighted Title - True Entities")
		title_true_entities = [e for e in dataset[selected_document]["entities"] if e["location"] == "title"]
		true_highlighted_title = highlight_text(document_data["title"], title_true_entities, color_map)
		st.markdown(true_highlighted_title, unsafe_allow_html=True)

		st.markdown("### Highlighted Title - Predicted Entities")
		title_predicted_entities = [e for e in predictions[selected_document]["entities"] if e["location"] == "title"]
		predicted_highlighted_title = highlight_text(document_data["title"], title_predicted_entities, color_map)
		st.markdown(predicted_highlighted_title, unsafe_allow_html=True)

		st.markdown("### Highlighted Title - Differences")
		title_differences = calculate_differences(title_true_entities, title_predicted_entities)
		differences_highlighted_title = highlight_text(document_data["title"], title_differences, color_map)
		st.markdown(differences_highlighted_title, unsafe_allow_html=True)

	# Highlight and display the abstract with true entities
	if "abstract" in document_data:
		st.markdown("### Highlighted Abstract - True Entities")
		abstract_true_entities = [e for e in dataset[selected_document]["entities"] if e["location"] == "abstract"]
		true_highlighted_abstract = highlight_text(document_data["abstract"], abstract_true_entities, color_map)
		st.markdown(true_highlighted_abstract, unsafe_allow_html=True)

		st.markdown("### Highlighted Abstract - Predicted Entities")
		abstract_predicted_entities = [
			e for e in predictions[selected_document]["entities"] if e["location"] == "abstract"
		]
		predicted_highlighted_abstract = highlight_text(
			document_data["abstract"], abstract_predicted_entities, color_map
		)
		st.markdown(predicted_highlighted_abstract, unsafe_allow_html=True)

		st.markdown("### Highlighted Abstract - Differences")
		abstract_differences = calculate_differences(abstract_true_entities, abstract_predicted_entities)
		differences_highlighted_abstract = highlight_text(document_data["abstract"], abstract_differences, color_map)
		st.markdown(differences_highlighted_abstract, unsafe_allow_html=True)

else:
	st.write("Please upload both Predictions and Dataset JSON files to proceed.")
