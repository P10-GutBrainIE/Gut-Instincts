import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, with_weights: bool = False):
        self.data = data
        self.with_weights = with_weights

        first_label = self.data[0]["labels"]
        if isinstance(first_label, (list, tuple)):
            self.task_type = "sequence"
        else:
            self.task_type = "classification"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample["input_ids"] = torch.as_tensor(sample["input_ids"], dtype=torch.long)
        sample["attention_mask"] = torch.as_tensor(sample["attention_mask"], dtype=torch.long)

        if self.task_type == "sequence":
            sample["labels"] = torch.as_tensor(sample["labels"], dtype=torch.long)
        elif self.task_type == "classification":
            sample["labels"] = torch.as_tensor(sample["labels"], dtype=torch.long).squeeze()

        if self.with_weights:
            sample["weight"] = torch.as_tensor(sample["weight"], dtype=torch.float)

        return sample
