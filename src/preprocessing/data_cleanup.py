import json
import os

def remove_documents_under_threshold(path, threshold, type):
    """
    Removes documents that are under the threshold and saves the remaining ones in a new JSON file in data/preprocessed

    Args:
        path (string): File path to the dataset
        threshold (int): Minimum number of entities or relations required
        type (string): Should be either "entities" or "relations"
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    filtered_data = {paper_id: content for paper_id, content in data.items() if len(content.get(type, [])) >= threshold}
    
    output_filename = os.path.basename(path).replace(".json", f"_removed_{type}_threshold_{threshold}.json")
    output_path = os.path.join("data", "preprocessed", output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"Filtered dataset saved to {output_path}")


def clean_bronze_text_spans():
    """
    Manually cleans bronze entities with c

    Args:
        path (string): File path to the dataset
        threshold (int): Minimum number of entities or relations required
        type (string): Should be either "entities" or "relations"
    """
    return


def remove_bronze_text_spans():
    return




if __name__ == "__main__":
    bronze_path = os.path.join("data", "Annotations", "Train", "bronze_quality", "json_format", "train_bronze.json")
    remove_documents_under_threshold(bronze_path, 1, "relations")

