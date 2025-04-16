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
	elif method == "multistep":
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer, milestones=lr_scheduler_dict["milestones"], gamma=lr_scheduler_dict["gamma"]
		)

	return scheduler
