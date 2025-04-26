import torch


def lr_scheduler(lr_scheduler_dict: dict, optimizer) -> torch.optim.lr_scheduler:
	method = lr_scheduler_dict["lr_scheduler"]["method"]
	if method == "cosine annealing":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer=optimizer,
			T_max=lr_scheduler_dict["num_epochs"],
			eta_min=lr_scheduler_dict["lr_scheduler"]["min_learning_rate"],
		)
	elif method == "reduce on plateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer=optimizer,
			factor=lr_scheduler_dict["lr_scheduler"]["factor"],
			patience=lr_scheduler_dict["lr_scheduler"]["patience"],
			threshold=lr_scheduler_dict["lr_scheduler"]["threshold"],
		)
	elif method == "one cycle":
		scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer=optimizer,
			max_lr=lr_scheduler_dict["lr_scheduler"]["max_learning_rate"],
			steps_per_epoch=1,
			epochs=lr_scheduler_dict["num_epochs"],
			pct_start=lr_scheduler_dict.get("pct_start", 0.3),
			anneal_strategy=lr_scheduler_dict.get("anneal_strategy", "cos"),
			div_factor=lr_scheduler_dict.get("div_factor", 25),
			final_div_factor=lr_scheduler_dict.get("final_div_factor", 1e4),
		)
	elif method == "custom":

		def custom_schedule(epoch):
			schedule = lr_scheduler_dict["custom_schedule"]
			for start, end, multiplier in schedule:
				if start <= epoch <= end:
					return multiplier

			epochs_post_warmup = epoch - (schedule[-1][1] + 1)

			return schedule[-1][2] * lr_scheduler_dict["lr_scheduler"]["gamma"] ** (
				epochs_post_warmup // lr_scheduler_dict["lr_scheduler"]["step_size"]
			)

		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: custom_schedule(epoch))

	return scheduler
