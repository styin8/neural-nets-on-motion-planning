import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os


class CustomDataset(Dataset):
    def __init__(self, opt, mode):
        self.opt = opt
        files = glob.glob(os.path.join(
            opt.dataroot, 'output_line_*.txt'))
        files.sort()
        l = len(files)
        if l == 0:
            raise Exception("No files found in dataroot")

        self.files = files
        if mode == "train":
            self.files = files[:int(
                l*self.opt.train_test_split*self.opt.train_val_split)]
        elif mode == "val":
            self.files = files[int(
                l*self.opt.train_test_split*self.opt.train_val_split):int(l*self.opt.train_test_split)]
        elif mode == "test":
            self.files = files[int(l*self.opt.train_test_split):]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], 'r') as f:
            data = list(map(float, f.readline().split()))
            inputs = torch.tensor(
                data[:self.opt.dof], dtype=torch.float32)
            outputs = torch.tensor(
                data[self.opt.dof:], dtype=torch.float32)
        return inputs, outputs


if __name__ == "__main__":
    opt = {
        "dataroot": "T:\Workspace\Master\\neural-nets-on-motion-planning\datasets"
    }
    dataset = CustomDataset(opt, mode="train")
    for item in dataset:
        print(item)
        break
    # print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for inputs, outputs in dataloader:
    #     print(inputs)
    #     print(outputs)
    #     break
