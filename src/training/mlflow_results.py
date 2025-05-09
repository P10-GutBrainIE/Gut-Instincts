import re
import mlflow
from mlflow.tracking import MlflowClient


def print_top_n_experiments(experiment_name: str = None, top_n: int = 20, task="ner"):
	"""
	Find and print the top runs with the highest F1_micro scores for an MLflow experiment.

	Retrieves runs for the given experiment (or all experiments if not specified), extracts the best F1_micro
	score for each run, and prints a summary table of the top runs sorted by F1_micro.

	Args:
	    experiment_name (str, optional): Name of the MLflow experiment. If None, considers all experiments.
	    top_n (int, optional): Number of top runs to display. Defaults to 20.
	"""
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

	runs = mlflow.search_runs(experiment_ids=experiment_ids, filter_string="")
	results = []

	if task == "re":
		runs = [run for _, run in runs.iterrows() if run.get("params.subtask", False)]
	elif task == "ner":
		runs = [
			run
			for _, run in runs.iterrows()
			if run.get("params.model_type") in ["huggingface", "bertlstmcrf", "bertdensecrf"]
		]
	else:
		raise ValueError(f"Unknown task: {task}")

	for run in runs:
		run_id = run["run_id"]
		try:
			metric_history = client.get_metric_history(run_id, "F1_micro")
			if metric_history:
				best_f1 = max(point.value for point in metric_history)
			else:
				best_f1 = float("-inf")

			weights = run.get("params.dataset_weights") if run.get("params.dataset_weights") else "No weights"
			qualities = run.get("params.dataset_qualities") if run.get("params.dataset_qualities") else "Not logged"
			qualities = str(re.findall(r"(de|[pgsb])", qualities))
			model_name = run.get("params.model_name") if run.get("params.model_name") else "Not logged"
			model_name = model_name.split("/")[-1]
			experiment_name = run.get("params.experiment_name") if run.get("params.experiment_name") else "Not logged"
			seed = run.get("params.seed") if run.get("params.seed") else "No"
			if task == "re":
				multiplier = (
					run.get("params.negative_sample_multiplier")
					if run.get("params.negative_sample_multiplier")
					else "Not logged"
				)

				results.append((experiment_name, model_name, f"{qualities} x {multiplier}", weights, seed, best_f1))
			else:
				results.append((experiment_name, model_name, qualities, weights, seed, best_f1))
		except Exception as e:
			print(f"Error processing run {run_id}: {e}")

	results.sort(key=lambda x: x[5], reverse=True)

	print(f"Best F1_micro values for {experiments_text}:")
	print(
		f"{'No.':<5} {'Experiment Name':<42} {'Model Name':<54} {'Qualities':<34} {'Weights':<30} {'Seed':<6} {'F1_micro':<8}"
	)
	print("-" * 180)
	for i, (experiment_name, model_name, qualities, weights, seed, best_f1) in enumerate(results[:top_n], start=1):
		print(
			f"{i:<5} {experiment_name:<42} {model_name:<54} {qualities:<34} {str(weights):<30} {seed:<6} {best_f1:<8.4f}"
		)


if __name__ == "__main__":
	print_top_n_experiments(experiment_name=None, top_n=100, task="ner")
