import mlflow
from mlflow.tracking import MlflowClient


def find_best_f1_for_experiment(experiment_name: str = None, top_n: int = 20):
	client = MlflowClient()

	if experiment_name is not None:
		experiment = mlflow.get_experiment_by_name(experiment_name)
		if not experiment:
			raise ValueError(f"Experiment '{experiment_name}' not found!")
		experiment_ids = [experiment.experiment_id]
		experiments_text = f"experiment '{experiment_name}'"
	else:
		all_experiments = list(mlflow.search_experiments())
		experiment_ids = [exp.experiment_id for exp in all_experiments]
		experiments_text = "all experiments"

	runs = mlflow.search_runs(experiment_ids=experiment_ids, filter_string="", output_format="list")
	results = []

	for run in runs:
		run_id = run.info.run_id
		f1_history = client.get_metric_history(run_id, "F1_micro")
		if not f1_history:
			continue
		best_f1 = max(metric.value for metric in f1_history)
		weights = run.data.params.get("dataset_weights", "Not logged")
		qualities = run.data.params.get("dataset_qualities", "Not logged")
		model_name = run.data.params.get("model_name", "Not logged").split("/")[-1]
		results.append((model_name, qualities, weights, best_f1))

	results.sort(key=lambda x: x[3], reverse=True)

	print(f"Best F1_micro values for {experiments_text}:")
	print(f"{'No.':<5} {'Model Name':<52} {'Qualities':<42} {'Weights':<26} {'F1_micro':<8}")
	print("-" * 140)
	for i, (model_name, qualities, weights, best_f1) in enumerate(results[:top_n], start=1):
		print(f"{i:<5} {model_name:<52} {qualities:<42} {weights:<26} {best_f1:<8.4f}")


if __name__ == "__main__":
	find_best_f1_for_experiment(experiment_name=None, top_n=10)
