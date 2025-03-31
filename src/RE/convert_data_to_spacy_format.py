import spacy
from spacy.tokens import Doc, DocBin, Span
from pathlib import Path
import os
import json
from utils.utils import load_json_data

if not Doc.has_extension("rel"):
    Doc.set_extension("rel", default={})

nlp = spacy.load("en_core_web_trf")


def convert_data_to_spacy_format(file_path: str, output_path: str):
    data = load_json_data(file_path)
    doc_bin = DocBin()

    for doc_id, doc_content in data.items():
        title = doc_content["metadata"]["title"]
        abstract = doc_content["metadata"]["abstract"]
        text = title + " " + abstract

        # Calculate offsets for title and abstract
        title_offset = 0
        abstract_offset = len(title) + 1

        # Create a spaCy doc object
        doc = nlp.make_doc(text)

        entities = []
        span_lookup = {}

        # Add entities to the doc
        for entity in doc_content["entities"]:
            loc = entity["location"]
            label = entity["label"]
            text_span = entity["text_span"]

            if loc == "title":
                start_idx = entity["start_idx"]
                end_idx = entity["end_idx"] + 1
            else:
                # TODO: check if this logic is correct
                start_idx = entity["start_idx"] + len(title) + 1
                end_idx = entity["end_idx"] + len(title) + 1 + 1
            #start_idx = entity["start_idx"] + (title_offset if loc == "title" else abstract_offset)
            #end_idx = entity["end_idx"] + (title_offset if loc == "title" else abstract_offset) + 1
            
            span = doc.char_span(start_idx, end_idx, label=label, alignment_mode="expand")
            
            if span is None:
                print("Skipping entity")
            else:
                entities.append(span)
                span_lookup[(start_idx, end_idx)] = span

        doc.ents = entities
        # dictionary to store relations, keyed by subject and object start and end indices
        doc._.rel = {}

        #for relation in doc_content.get("relations", []):
        for relation in doc_content["relations"]:
            subject_loc = relation["subject_location"]
            object_loc = relation["object_location"]
            subject_start_idx = relation["subject_start_idx"] + (title_offset if subject_loc == "title" else abstract_offset)
            subject_end_idx = relation["subject_end_idx"] + (title_offset if subject_loc == "title" else abstract_offset)
            object_start_idx = relation["object_start_idx"] + (title_offset if object_loc == "title" else abstract_offset)
            object_end_idx = relation["object_end_idx"] + (title_offset if object_loc == "title" else abstract_offset)
            predicate = relation["predicate"]

            subject_key = (subject_start_idx, subject_end_idx)
            object_key = (object_start_idx, object_end_idx)

            if subject_key in span_lookup and object_key in span_lookup:
                subject_span = span_lookup[subject_key]
                object_span = span_lookup[object_key]
                subject_tuple = (subject_span.start, subject_span.end)
                object_tuple = (object_span.start, object_span.end)

                if subject_tuple not in doc._.rel:
                    doc._.rel[subject_tuple] = {}
                doc._.rel[subject_tuple][object_tuple] = {"predicate": predicate}

        doc_bin.add(doc)
    
    doc_bin.to_disk(output_path)
    print(f"Saved {len(doc_bin)} documents to {output_path}")


if __name__ == "__main__":
    shared_path = os.path.join("data", "Annotations", "Train")

    input_files = [
        os.path.join(shared_path, "bronze_quality", "json_format", "train_bronze.json"),
        os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
        os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
        os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
        os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
    ]

    output_files = [
        os.path.join("src", "RE", "output", "train_bronze.spacy"),
        os.path.join("src", "RE", "output", "train_silver.spacy"),
        os.path.join("src", "RE", "output", "train_gold.spacy"),
        os.path.join("src", "RE", "output", "train_platinum.spacy"),
        os.path.join("src", "RE", "output", "dev.spacy"),
    ]
    
    convert_data_to_spacy_format(input_files[3], output_files[3])