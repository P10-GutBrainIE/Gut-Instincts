def freeze_bert(model):
	"""
	Freezes the BERT encoder and embeddings regardless of wrapper class.
	"""
	# HuggingFace TokenClassification models
	if hasattr(model, "model"):
		base = getattr(model.model, "bert", None) or getattr(model.model, "base_model", None)
	else:
		# BERTLSTMCRF
		base = getattr(model, "bert", None)

	if base is None:
		raise ValueError("Could not find BERT model to freeze.")

	for param in base.parameters():
		param.requires_grad = False

	print("--- Checking requires_grad after freezing ---")
	for name, param in model.named_parameters():
		print(f"{name:60} | {str(param.shape):20} | requires_grad={param.requires_grad}")
	n_total = sum(1 for _ in model.parameters())
	n_trainable = sum(p.requires_grad for p in model.parameters())
	print(f"Trainable parameters: {n_trainable} / {n_total}")


def unfreeze_bert(model):
	"""
	Unfreezes the BERT encoder and embeddings regardless of wrapper class.
	"""
	if hasattr(model, "model"):
		base = getattr(model.model, "bert", None) or getattr(model.model, "base_model", None)
	else:
		base = getattr(model, "bert", None)

	if base is None:
		raise ValueError("Could not find BERT model to unfreeze.")

	for param in base.parameters():
		param.requires_grad = True
