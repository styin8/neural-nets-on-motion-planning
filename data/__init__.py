from data.custom_datasets import CustomDataset
import torch


class CustomDatasetDataLoader():
    def __init__(self, opt, mode):
        self.opt = opt
        self.dataset = CustomDataset(opt, mode)
        print(f"dataset {type(self.dataset).__name__} {mode}was created!")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size if mode == "train" else 1,
            shuffle=True if mode == "train" else False,
            num_workers=int(opt.num_threads)
        )

    def __iter__(self):
        for data, label in self.dataloader:
            yield data, label

    def __len__(self):
        return len(self.dataset)


def create_dataset(opt, mode):

    dataset_loader = CustomDatasetDataLoader(opt, mode)

    return dataset_loader
