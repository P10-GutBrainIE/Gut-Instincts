import mlflow


def print_top_n_experiments(experiment_name: str = None, top_n: int = 20):
	"""
	Find and print the top runs with the highest F1_micro scores for an MLflow experiment.

	Retrieves runs for the given experiment (or all experiments if not specified), extracts the best F1_micro
	score for each run, and prints a summary table of the top runs sorted by F1_micro.

	Args:
	    experiment_name (str, optional): Name of the MLflow experiment. If None, considers all experiments.
	    top_n (int, optional): Number of top runs to display. Defaults to 20.
	"""
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
		best_f1 = run.data.metrics.get("Best F1_micro", 0)
		weights = run.data.params.get("dataset_weights", "No weights")
		qualities = run.data.params.get("dataset_qualities", "Not logged")
		remove_html = run.data.params.get("remove_html", "Not logged")
		model_name = run.data.params.get("model_name", "Not logged").split("/")[-1]
		results.append((model_name, qualities, weights, remove_html, best_f1))

	results.sort(key=lambda x: x[4], reverse=True)

	print(f"Best F1_micro values for {experiments_text}:")
	print(f"{'No.':<5} {'Model Name':<52} {'Qualities':<42} {'Weights':<26} {'Remove HTML':<12} {'F1_micro':<8}")
	print("-" * 152)
	for i, (model_name, qualities, weights, remove_html, best_f1) in enumerate(results[:top_n], start=1):
		print(f"{i:<5} {model_name:<52} {qualities:<42} {weights:<26} {remove_html:<12} {best_f1:<8.4f}")


if __name__ == "__main__":
	print_top_n_experiments(experiment_name=None, top_n=10)
