import torch


class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, is_validation: bool = False):
		self.data = data
		self.is_validation = is_validation

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long)
		sample["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long)
		sample["labels"] = torch.tensor(sample["labels"], dtype=torch.long)
		if not self.is_validation:
			sample["weight"] = torch.tensor(sample["weight"], dtype=torch.float)
		return sample
