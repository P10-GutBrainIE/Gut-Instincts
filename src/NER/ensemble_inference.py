import os
import json
import argparse
import yaml
from NER.inference import NERInference

def load_model_predictions(config):
    predictions_per_model = []
    for model in config["model_paths"]:

        ner_inference = NERInference(
            os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
            model_name_path=os.path.join("models", f"{config['experiment_name']}"),
            save_path=os.path.join("data_inference_results", f"{config['experiment_name']}.json"),
	    )

        pass

def majority_vote():
    pass

def save_ensemble_results(predictions, save_path):
    with open(save_path, "w") as file:
        json.dump(predictions, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        os.makedirs("models", exist_ok=True)

    predictions = load_model_predictions(config)
    #ensemble_predictions = majority_vote(predictions)
    #save_ensemble_results(predictions= ensemble_predictions, save_path = os.path.join("data_inference_results", f"{config['experiment_name']}.json"))

