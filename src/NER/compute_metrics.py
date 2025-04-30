import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from NER.inference import NERInference
from utils.utils import load_entity_labels, load_relation_labels, load_json_data


def compute_metrics(predictions, labels):
	true_predictions = [p for pred, lbl in zip(predictions, labels) for p, la in zip(pred, lbl) if la != -100]
	true_labels = [la for pred, lbl in zip(predictions, labels) for _, la in zip(pred, lbl) if la != -100]

	true_predictions_no_o = [
		p for pred, lbl in zip(predictions, labels) for p, la in zip(pred, lbl) if la not in [-100, 0]
	]
	true_labels_no_o = [la for pred, lbl in zip(predictions, labels) for _, la in zip(pred, lbl) if la not in [-100, 0]]

	metrics = {}
	log_metrics = {}
	for name, (preds, lbls) in {
		"all": (true_predictions, true_labels),
		"no_o": (true_predictions_no_o, true_labels_no_o),
	}.items():
		accuracy = accuracy_score(lbls, preds)
		precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
			lbls, preds, average="micro", zero_division=0
		)
		precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
			lbls, preds, average="macro", zero_division=0
		)
		total = accuracy + precision_micro + recall_micro + f1_micro + precision_macro + recall_macro + f1_macro

		metrics[name] = {
			"Accuracy": accuracy,
			"Precision_micro": precision_micro,
			"Recall_micro": recall_micro,
			"F1_micro": f1_micro,
			"Precision_macro": precision_macro,
			"Recall_macro": recall_macro,
			"F1_macro": f1_macro,
			"Total": total,
		}

		log_metrics.update(
			{
				f"{name}_accuracy": accuracy,
				f"{name}_precision_micro": precision_micro,
				f"{name}_recall_micro": recall_micro,
				f"{name}_f1_micro": f1_micro,
				f"{name}_precision_macro": precision_macro,
				f"{name}_recall_macro": recall_macro,
				f"{name}_f1_macro": f1_macro,
			}
		)

	return metrics, log_metrics


def compute_evaluation_metrics(model, model_name, model_type, subtask):
	model.to("cpu")

	if model_type == "re":
		from NER.inference import REInference

		re_inference = REInference(
			test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
			ner_predictions_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
			model_name=model_name,
			model_type=model_type,
			validation_model=model,
			subtask=subtask,
		)
		inference_results = re_inference.perform_inference()

		if subtask == "6.2.1":
			precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = (
				RE_evaluation_subtask_621(inference_results)
			)
		elif subtask == "6.2.2":
			precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = (
				RE_evaluation_subtask_622(inference_results)
			)
		elif subtask == "6.2.3":
			precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = (
				RE_evaluation_subtask_623(inference_results)
			)
		else:
			return ValueError("No matching subtask")

		return {
			"Precision_micro": precision_micro,
			"Recall_micro": recall_micro,
			"F1_micro": f1_micro,
			"Precision_macro": precision_macro,
			"Recall_macro": recall_macro,
			"F1_macro": f1_macro,
		}

	else:
		ner_inference = NERInference(
			test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
			model_name=model_name,
			model_type=model_type,
			validation_model=model,
		)

		inference_results = ner_inference.perform_inference()
		precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = NER_evaluation(
			inference_results
		)

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


def RE_evaluation_subtask_621(predictions):
	ground_truth_binary_tag_RE = dict()
	count_annotated_relations_per_label = {}
	entity_labels = load_entity_labels()[1:]
	ground_truth = load_json_data(file_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	for pmid, article in ground_truth.items():
		if pmid not in ground_truth_binary_tag_RE:
			ground_truth_binary_tag_RE[pmid] = []
		for relation in article["binary_tag_based_relations"]:
			subject_label = str(relation["subject_label"])
			object_label = str(relation["object_label"])

			label = (subject_label, object_label)
			ground_truth_binary_tag_RE[pmid].append(label)

			if label not in count_annotated_relations_per_label:
				count_annotated_relations_per_label[label] = 0
			count_annotated_relations_per_label[label] += 1

	count_predicted_relations_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}
	count_true_positives_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}

	for pmid in predictions.keys():
		try:
			relations = predictions[pmid]["binary_tag_based_relations"]
		except KeyError:
			raise KeyError(f'{pmid} - Not able to find field "binary_tag_based_relations" within article')

		for relation in relations:
			try:
				subject_label = str(relation["subject_label"])
				object_label = str(relation["object_label"])
			except KeyError:
				raise KeyError(f"{pmid} - Not able to find one or more of the expected fields for relation: {relation}")

			if subject_label not in entity_labels:
				raise NameError(f"{pmid} - Illegal subject entity label {subject_label} for relation: {relation}")

			if object_label not in entity_labels:
				raise NameError(f"{pmid} - Illegal object entity label {object_label} for relation: {relation}")

			label = (subject_label, object_label)
			if label in count_predicted_relations_per_label:
				count_predicted_relations_per_label[label] += 1

			if label in ground_truth_binary_tag_RE[pmid]:
				count_true_positives_per_label[label] += 1

	count_annotated_relations = sum(
		count_annotated_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)
	count_predicted_relations = sum(
		count_predicted_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)
	count_true_positives = sum(
		count_true_positives_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)

	micro_precision = count_true_positives / (count_predicted_relations + 1e-10)
	micro_recall = count_true_positives / (count_annotated_relations + 1e-10)
	micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

	precision, recall, f1 = 0, 0, 0
	n = 0
	for label in list(count_annotated_relations_per_label.keys()):
		n += 1
		current_precision = count_true_positives_per_label[label] / (count_predicted_relations_per_label[label] + 1e-10)
		current_recall = count_true_positives_per_label[label] / (count_annotated_relations_per_label[label] + 1e-10)

		precision += current_precision
		recall += current_recall
		f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))

	precision = precision / n
	recall = recall / n
	f1 = f1 / n

	return precision, recall, f1, micro_precision, micro_recall, micro_f1


def RE_evaluation_subtask_622(predictions):
	ground_truth_ternary_tag_RE = dict()
	count_annotated_relations_per_label = {}
	entity_labels = load_entity_labels()[1:]
	relation_labels = load_relation_labels()[1:]
	ground_truth = load_json_data(file_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	for pmid, article in ground_truth.items():
		if pmid not in ground_truth_ternary_tag_RE:
			ground_truth_ternary_tag_RE[pmid] = []
		for relation in article["ternary_tag_based_relations"]:
			subject_label = str(relation["subject_label"])
			predicate = str(relation["predicate"])
			object_label = str(relation["object_label"])

			label = (subject_label, predicate, object_label)
			ground_truth_ternary_tag_RE[pmid].append(label)

			if label not in count_annotated_relations_per_label:
				count_annotated_relations_per_label[label] = 0
			count_annotated_relations_per_label[label] += 1

	count_predicted_relations_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}
	count_true_positives_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}

	for pmid in predictions.keys():
		try:
			relations = predictions[pmid]["ternary_tag_based_relations"]
		except KeyError:
			raise KeyError(f'{pmid} - Not able to find field "ternary_tag_based_relations" within article')

		for relation in relations:
			try:
				subject_label = str(relation["subject_label"])
				predicate = str(relation["predicate"])
				object_label = str(relation["object_label"])
			except KeyError:
				raise KeyError(f"{pmid} - Not able to find one or more of the expected fields for relation: {relation}")

			if subject_label not in entity_labels:
				raise NameError(f"{pmid} - Illegal subject entity label {subject_label} for relation: {relation}")

			if object_label not in entity_labels:
				raise NameError(f"{pmid} - Illegal object entity label {object_label} for relation: {relation}")

			if predicate not in relation_labels:
				raise NameError(f"{pmid} - Illegal predicate {predicate} for relation: {relation}")

			label = (subject_label, predicate, object_label)
			if label in count_predicted_relations_per_label:
				count_predicted_relations_per_label[label] += 1

			if label in ground_truth_ternary_tag_RE[pmid]:
				count_true_positives_per_label[label] += 1

	count_annotated_relations = sum(
		count_annotated_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)
	count_predicted_relations = sum(
		count_predicted_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)
	count_true_positives = sum(
		count_true_positives_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)

	micro_precision = count_true_positives / (count_predicted_relations + 1e-10)
	micro_recall = count_true_positives / (count_annotated_relations + 1e-10)
	micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

	precision, recall, f1 = 0, 0, 0
	n = 0
	for label in list(count_annotated_relations_per_label.keys()):
		n += 1
		current_precision = count_true_positives_per_label[label] / (count_predicted_relations_per_label[label] + 1e-10)
		current_recall = count_true_positives_per_label[label] / (count_annotated_relations_per_label[label] + 1e-10)

		precision += current_precision
		recall += current_recall
		f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))

	precision = precision / n
	recall = recall / n
	f1 = f1 / n

	return precision, recall, f1, micro_precision, micro_recall, micro_f1


def RE_evaluation_subtask_623(predictions):
	ground_truth_ternary_mention_RE = dict()
	count_annotated_relations_per_label = {}
	entity_labels = load_entity_labels()[1:]
	relation_labels = load_relation_labels()[1:]
	ground_truth = load_json_data(file_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	for pmid, article in ground_truth.items():
		if pmid not in ground_truth_ternary_mention_RE:
			ground_truth_ternary_mention_RE[pmid] = []
		for relation in article["ternary_mention_based_relations"]:
			subject_text_span = str(relation["subject_text_span"])
			subject_label = str(relation["subject_label"])
			predicate = str(relation["predicate"])
			object_text_span = str(relation["object_text_span"])
			object_label = str(relation["object_label"])

			entry = (subject_text_span, subject_label, predicate, object_text_span, object_label)
			ground_truth_ternary_mention_RE[pmid].append(entry)

			label = (subject_label, predicate, object_label)
			if label not in count_annotated_relations_per_label:
				count_annotated_relations_per_label[label] = 0
			count_annotated_relations_per_label[label] += 1

	count_predicted_relations_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}
	count_true_positives_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}

	for pmid in predictions.keys():
		try:
			relations = predictions[pmid]["ternary_mention_based_relations"]
		except KeyError:
			raise KeyError(f'{pmid} - Not able to find field "ternary_mention_based_relations" within article')

		for relation in relations:
			try:
				subject_text_span = str(relation["subject_text_span"])
				subject_label = str(relation["subject_label"])
				predicate = str(relation["predicate"])
				object_text_span = str(relation["object_text_span"])
				object_label = str(relation["object_label"])
			except KeyError:
				raise KeyError(f"{pmid} - Not able to find one or more of the expected fields for relation: {relation}")

			if subject_label not in entity_labels:
				raise NameError(f"{pmid} - Illegal subject entity label {subject_label} for relation: {relation}")

			if object_label not in entity_labels:
				raise NameError(f"{pmid} - Illegal object entity label {object_label} for relation: {relation}")

			if predicate not in relation_labels:
				raise NameError(f"{pmid} - Illegal predicate {predicate} for relation: {relation}")

			entry = (subject_text_span, subject_label, predicate, object_text_span, object_label)
			label = (subject_label, predicate, object_label)

			if label in count_predicted_relations_per_label:
				count_predicted_relations_per_label[label] += 1

			if entry in ground_truth_ternary_mention_RE[pmid]:
				count_true_positives_per_label[label] += 1

	count_annotated_relations = sum(
		count_annotated_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)
	count_predicted_relations = sum(
		count_predicted_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)
	count_true_positives = sum(
		count_true_positives_per_label[label] for label in list(count_annotated_relations_per_label.keys())
	)

	micro_precision = count_true_positives / (count_predicted_relations + 1e-10)
	micro_recall = count_true_positives / (count_annotated_relations + 1e-10)
	micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

	precision, recall, f1 = 0, 0, 0
	n = 0
	for label in list(count_annotated_relations_per_label.keys()):
		n += 1
		current_precision = count_true_positives_per_label[label] / (count_predicted_relations_per_label[label] + 1e-10)
		current_recall = count_true_positives_per_label[label] / (count_annotated_relations_per_label[label] + 1e-10)

		precision += current_precision
		recall += current_recall
		f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))

	precision = precision / n
	recall = recall / n
	f1 = f1 / n

	return precision, recall, f1, micro_precision, micro_recall, micro_f1
