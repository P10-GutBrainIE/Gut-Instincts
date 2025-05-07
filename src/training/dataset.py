import torch


class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, with_weights: bool = False):
		self.data = data
		self.with_weights = with_weights

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		sample["input_ids"] = torch.as_tensor(sample["input_ids"], dtype=torch.long)
		sample["attention_mask"] = torch.as_tensor(sample["attention_mask"], dtype=torch.long)
		sample["labels"] = torch.as_tensor(sample["labels"], dtype=torch.long)

		if self.with_weights:
			sample["weight"] = torch.as_tensor(sample["weight"], dtype=torch.float)

		return sample
