import os
from NER.inference import NERInference
from utils.utils import load_entity_labels, load_json_data


def compute_metrics(model, model_name, model_type, remove_html):
	ner_inference = NERInference(
		test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		model_name=model_name,
		model_type=model_type,
		validation_model=model,
		remove_html=remove_html,
	)

	inference_results = ner_inference.perform_inference()
	precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = NER_evaluation(inference_results)

	return {
		"Precision_micro": precision_micro,
		"Recall_micro": recall_micro,
		"F1_micro": f1_micro,
		"Precision_macro": precision_macro,
		"Recall_macro": recall_macro,
		"F1_macro": f1_macro,
	}


def NER_evaluation(predictions):
	ground_truth_NER = dict()
	count_annotated_entities_per_label = {}
	entity_labels = load_entity_labels()[1:]
	ground_truth = load_json_data(file_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	for pmid, article in ground_truth.items():
		if pmid not in ground_truth_NER:
			ground_truth_NER[pmid] = []
		for entity in article["entities"]:
			start_idx = int(entity["start_idx"])
			end_idx = int(entity["end_idx"])
			location = str(entity["location"])
			text_span = str(entity["text_span"])
			label = str(entity["label"])

			entry = (start_idx, end_idx, location, text_span.lower(), label)
			ground_truth_NER[pmid].append(entry)

			if label not in count_annotated_entities_per_label:
				count_annotated_entities_per_label[label] = 0
			count_annotated_entities_per_label[label] += 1

	count_predicted_entities_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
	count_true_positives_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}

	for pmid in predictions.keys():
		try:
			entities = predictions[pmid]["entities"]
		except KeyError:
			raise KeyError(f'{pmid} - Not able to find field "entities" within article')

		for entity in entities:
			try:
				start_idx = int(entity["start_idx"])
				end_idx = int(entity["end_idx"])
				location = str(entity["location"])
				text_span = str(entity["text_span"])
				label = str(entity["label"])
			except KeyError:
				raise KeyError(f"{pmid} - Not able to find one or more of the expected fields for entity: {entity}")

			if label not in entity_labels:
				raise NameError(f"{pmid} - Illegal label {label} for entity: {entity}")

			if label in count_predicted_entities_per_label:
				count_predicted_entities_per_label[label] += 1

			entry = (start_idx, end_idx, location, text_span, label)
			if entry in ground_truth_NER[pmid]:
				count_true_positives_per_label[label] += 1

	count_annotated_entities = sum(
		count_annotated_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys())
	)
	count_predicted_entities = sum(
		count_predicted_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys())
	)
	count_true_positives = sum(
		count_true_positives_per_label[label] for label in list(count_annotated_entities_per_label.keys())
	)

	micro_precision = count_true_positives / (count_predicted_entities + 1e-10)
	micro_recall = count_true_positives / (count_annotated_entities + 1e-10)
	micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

	precision, recall, f1 = 0, 0, 0
	n = 0
	for label in list(count_annotated_entities_per_label.keys()):
		n += 1
		current_precision = count_true_positives_per_label[label] / (count_predicted_entities_per_label[label] + 1e-10)
		current_recall = count_true_positives_per_label[label] / (count_annotated_entities_per_label[label] + 1e-10)

		precision += current_precision
		recall += current_recall
		f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))

	precision = precision / n
	recall = recall / n
	f1 = f1 / n

	return precision, recall, f1, micro_precision, micro_recall, micro_f1
