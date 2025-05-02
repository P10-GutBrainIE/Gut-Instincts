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
