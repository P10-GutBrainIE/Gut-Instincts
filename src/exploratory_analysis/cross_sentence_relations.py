import json
import re
import os


def sentence_spans(text):
	"""
	Splits the text into sentence spans by naive regex on '.', '?', '!'.
	Returns a list of (start_index, end_index) for each sentence.
	"""
	pattern = re.compile(r"[.?!]+")
	matches = list(pattern.finditer(text))
	spans = []
	start = 0
	for m in matches:
		end = m.end()
		spans.append((start, end))
		start = end
	if start < len(text):
		spans.append((start, len(text)))
	return spans


def is_cross_sentence(subject_start, subject_end, object_start, object_end, spans):
	"""
	Determines whether subject and object are in different sentences
	within the same text field (title or abstract).
	"""
	subject_sentence_index = None
	object_sentence_index = None
	for i, (s_start, s_end) in enumerate(spans):
		if s_start <= subject_start < s_end and s_start <= subject_end <= s_end:
			subject_sentence_index = i
		if s_start <= object_start < s_end and s_start <= object_end <= s_end:
			object_sentence_index = i
		if subject_sentence_index is not None and object_sentence_index is not None:
			break

	if subject_sentence_index is None or object_sentence_index is None:
		return False
	return subject_sentence_index != object_sentence_index


def mention_distance(s_start, s_end, o_start, o_end):
	"""
	Computes how far apart two mentions are, if they do not overlap.
	Distance is measured from the end of one mention to the start of the other
	if they occur sequentially; otherwise 0 if they overlap.
	"""
	if s_end < o_start:
		return o_start - s_end
	elif o_end < s_start:
		return s_start - o_end
	else:
		return 0


def analyze_file(filename, top_n=5):
	"""
	Analyzes a single JSON file for:
	  1) cross-sentence relations
	  2) 'farthest apart' relations in the same text field
	Returns:
	  total_relations, cross_sentence_count, large_gap_relations (a list)
	"""
	with open(filename, "r") as f:
		data = json.load(f)

	total_relations = 0
	cross_sentence_count = 0

	distance_info = []

	for doc_id, content in data.items():
		title = content["metadata"].get("title", "")
		abstract = content["metadata"].get("abstract", "")

		title_spans = sentence_spans(title)
		abstract_spans = sentence_spans(abstract)

		for relation in content.get("relations", []):
			total_relations += 1
			s_loc = relation["subject_location"]
			o_loc = relation["object_location"]
			s_start = relation["subject_start_idx"]
			s_end = relation["subject_end_idx"]
			o_start = relation["object_start_idx"]
			o_end = relation["object_end_idx"]

			if s_loc != o_loc:
				cross_sentence_count += 1
				continue

			if s_loc == "title":
				if is_cross_sentence(s_start, s_end, o_start, o_end, title_spans):
					cross_sentence_count += 1
			else:
				if is_cross_sentence(s_start, s_end, o_start, o_end, abstract_spans):
					cross_sentence_count += 1

			dist = mention_distance(s_start, s_end, o_start, o_end)
			if dist > 0:
				if s_loc == "title":
					sub_text = title[s_start:s_end]
					obj_text = title[o_start:o_end]
				else:
					sub_text = abstract[s_start:s_end]
					obj_text = abstract[o_start:o_end]

				distance_info.append((dist, doc_id, sub_text, obj_text, s_loc))

	distance_info.sort(key=lambda x: x[0], reverse=True)

	largest_gap_relations = distance_info[:top_n]

	return total_relations, cross_sentence_count, largest_gap_relations


# -------------------------
# MAIN: Analyze all 3 files
# -------------------------
shared_path = os.path.join("data", "Annotations", "Train")

files = [
	os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
	os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
	os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
]

grand_total_relations = 0
grand_cross_count = 0

top_results_by_file = {}

for file in files:
	total, cross_count, largest_gaps = analyze_file(file, top_n=5)
	grand_total_relations += total
	grand_cross_count += cross_count

	print(f"\nFile: {file}")
	print(f"  Total relations:            {total}")
	print(f"  Cross-sentence relations:   {cross_count}")
	print("  Sample of largest-gap relations (distance, doc_id, subject_text, object_text, location):")
	for item in largest_gaps:
		print("    ", item)

	top_results_by_file[file] = largest_gaps

print("\n----------------------------------------------")
print("Overall results across all three JSON files   ")
print("----------------------------------------------")
print(f"Grand total relations:          {grand_total_relations}")
print(f"Grand cross-sentence relations: {grand_cross_count}")
