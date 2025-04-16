import torch


def lr_scheduler(lr_scheduler_dict: dict, optimizer) -> torch.optim.lr_scheduler:
	method = lr_scheduler_dict["method"]
	if method == "cosine annealing":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer=optimizer,
			T_max=lr_scheduler_dict["num_epochs"],
			eta_min=lr_scheduler_dict["min_learning_rate"],
		)
	elif method == "reduce on plateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer=optimizer,
			factor=lr_scheduler_dict["factor"],
			patience=lr_scheduler_dict["patience"],
			threshold=lr_scheduler_dict["threshold"],
		)
	elif method == "custom":

		def custom_schedule(epoch, schedule=lr_scheduler_dict["custom_schedule"]):
			for start, end, multiplier in schedule:
				if start <= epoch <= end:
					return multiplier

		warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: custom_schedule(epoch))
		training = torch.optim.lr_scheduler.StepLR(
			optimizer, lr_scheduler_dict["step_size"], lr_scheduler_dict["gamma"]
		)
		scheduler = torch.optim.lr_scheduler.SequentialLR(
			optimizer, schedulers=[warmup, training], milestones=[lr_scheduler_dict["custom_schedule"][-1][1]]
		)

	return scheduler
