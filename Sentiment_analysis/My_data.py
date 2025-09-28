from torch.utils.data import Dataset
from datasets import load_from_disk

class My_dataset(Dataset):
    def __init__(self,splist):
        self.dataset=load_from_disk(r"G:\00AI-LM\05_基于 BERT 的中文评价情感分析2025年聚客大模型第三期\demo_5\data\ChnSentiCorp")
        if splist=="test":
            self.dataset = self.dataset["test"]
        elif splist=="train":
            self.dataset = self.dataset["train"]
        elif splist=="validation":
            self.Dataset = self.dataset["validation"]

    def __len__(self):
        return len(self.Dataset)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']

        return text, label

if __name__ == '__main__':
    dataset = My_dataset("test")
    for data in dataset:
        print(data)

