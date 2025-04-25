import mlflow
from mlflow.tracking import MlflowClient


def find_best_f1_for_experiment(experiment_name):
	experiment = mlflow.get_experiment_by_name(experiment_name)
	if not experiment:
		raise ValueError(f"Experiment '{experiment_name}' not found!")

	client = MlflowClient()
	runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="", output_format="list")
	results = []

	for run in runs:
		run_id = run.info.run_id
		f1_history = client.get_metric_history(run_id, "F1_micro")
		best_f1 = max(metric.value for metric in f1_history)
		weights = run.data.params.get("dataset_weights", "Not logged")
		qualities = run.data.params.get("dataset_qualities", "Not logged")
		results.append((best_f1, weights, qualities, run_id))

	results.sort(key=lambda x: x[0], reverse=True)

	print(f"Best F1_micro values for experiment '{experiment_name}':")
	print(f"{'F1_micro':<8} {'Weights':<26} {'Qualities':<42} {'Run ID':<34}")
	print("-" * 111)
	for best_f1, weights, qualities, run_id in results:
		print(f"{best_f1:<8.4f} {weights:<26} {qualities:<42} {run_id:<34}")


if __name__ == "__main__":
	find_best_f1_for_experiment("BioLinkBERT-base")
