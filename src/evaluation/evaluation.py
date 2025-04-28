import json
import os
import argparse
import yaml
from utils.utils import make_dataset_dir_name

parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
args = parser.parse_args()

with open(args.config, "r") as file:
	config = yaml.safe_load(file)

dataset_dir_name = make_dataset_dir_name(config)

# DEFINE HERE THE PATH(S) TO YOUR PREDICTIONS
PREDICTIONS_PATH_6_1 = os.path.join("data_inference_results", config["experiment_name"], f"{dataset_dir_name}.json")
PREDICTIONS_PATH_6_2 = "org_T621_BaselineRun_ATLOP.json"
PREDICTIONS_PATH_6_3 = "org_T622_BaselineRun_ATLOP.json"
PREDICTIONS_PATH_6_4 = "org_T623_BaselineRun_ATLOP.json"

# DEFINE HERE FOR WHICH SUBTASK(S) YOU WANT TO EVAL YOUR PREDICTIONS
eval_6_1_NER = True
eval_6_2_binary_tag_RE = False
eval_6_3_ternary_tag_RE = False
eval_6_4_ternary_mention_RE = False

GROUND_TRUTH_PATH = os.path.join("data", "Annotations", "Dev", "json_format", "dev.json")
# GROUND_TRUTH_PATH = os.path.join("data_preprocessed", "test_data.json")
try:
	with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
		ground_truth = json.load(file)
except OSError:
	raise OSError(f"Error in opening the specified json file: {GROUND_TRUTH_PATH}")

LEGAL_ENTITY_LABELS = [
	"anatomical location",
	"animal",
	"bacteria",
	"biomedical technique",
	"chemical",
	"DDF",
	"dietary supplement",
	"drug",
	"food",
	"gene",
	"human",
	"microbiome",
	"statistical technique",
]

LEGAL_RELATION_LABELS = [
	"administered",
	"affect",
	"change abundance",
	"change effect",
	"change expression",
	"compared to",
	"impact",
	"influence",
	"interact",
	"is a",
	"is linked to",
	"located in",
	"part of",
	"produced by",
	"strike",
	"target",
	"used by",
]


def eval_submission_6_1_NER(path):
	try:
		with open(path, "r", encoding="utf-8") as file:
			predictions = json.load(file)
	except OSError:
		raise OSError(f"Error in opening the specified json file: {path}")

	ground_truth_NER = dict()
	count_annotated_entities_per_label = {}

	for pmid, article in ground_truth.items():
		if pmid not in ground_truth_NER:
			ground_truth_NER[pmid] = []
		for entity in article["entities"]:
			start_idx = int(entity["start_idx"])
			end_idx = int(entity["end_idx"])
			location = str(entity["location"])
			text_span = str(entity["text_span"])
			label = str(entity["label"])

			entry = (start_idx, end_idx, location, text_span, label)
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

			if label not in LEGAL_ENTITY_LABELS:
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


def eval_submission_6_2_binary_tag_RE(path):
	try:
		with open(path, "r", encoding="utf-8") as file:
			predictions = json.load(file)
	except OSError:
		raise OSError(f"Error in opening the specified json file: {path}")

	ground_truth_binary_tag_RE = dict()
	count_annotated_relations_per_label = {}

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

			if subject_label not in LEGAL_ENTITY_LABELS:
				raise NameError(f"{pmid} - Illegal subject entity label {subject_label} for relation: {relation}")

			if object_label not in LEGAL_ENTITY_LABELS:
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


def eval_submission_6_3_ternary_tag_RE(path):
	try:
		with open(path, "r", encoding="utf-8") as file:
			predictions = json.load(file)
	except OSError:
		raise OSError(f"Error in opening the specified json file: {path}")

	ground_truth_ternary_tag_RE = dict()
	count_annotated_relations_per_label = {}

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

			if subject_label not in LEGAL_ENTITY_LABELS:
				raise NameError(f"{pmid} - Illegal subject entity label {subject_label} for relation: {relation}")

			if object_label not in LEGAL_ENTITY_LABELS:
				raise NameError(f"{pmid} - Illegal object entity label {object_label} for relation: {relation}")

			if predicate not in LEGAL_RELATION_LABELS:
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


def eval_submission_6_4_ternary_mention_RE(path):
	try:
		with open(path, "r", encoding="utf-8") as file:
			predictions = json.load(file)
	except OSError:
		raise OSError(f"Error in opening the specified json file: {path}")

	ground_truth_ternary_mention_RE = dict()
	count_annotated_relations_per_label = {}

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

			if subject_label not in LEGAL_ENTITY_LABELS:
				raise NameError(f"{pmid} - Illegal subject entity label {subject_label} for relation: {relation}")

			if object_label not in LEGAL_ENTITY_LABELS:
				raise NameError(f"{pmid} - Illegal object entity label {object_label} for relation: {relation}")

			if predicate not in LEGAL_RELATION_LABELS:
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


if __name__ == "__main__":
	round_to_decimal_position = 4

	if eval_6_1_NER:
		precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_1_NER(PREDICTIONS_PATH_6_1)
		print("\n\n=== 6_1_NER ===")
		print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
		print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
		print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
		print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
		print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
		print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

	if eval_6_2_binary_tag_RE:
		precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_2_binary_tag_RE(
			PREDICTIONS_PATH_6_2
		)
		print("\n\n=== 6_2_binary_tag_RE ===")
		print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
		print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
		print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
		print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
		print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
		print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

	if eval_6_3_ternary_tag_RE:
		precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_3_ternary_tag_RE(
			PREDICTIONS_PATH_6_3
		)
		print("\n\n=== 6_3_ternary_tag_RE ===")
		print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
		print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
		print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
		print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
		print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
		print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

	if eval_6_4_ternary_mention_RE:
		precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_4_ternary_mention_RE(
			PREDICTIONS_PATH_6_4
		)
		print("\n\n=== 6_4_ternary_mention_RE ===")
		print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
		print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
		print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
		print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
		print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
		print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")
