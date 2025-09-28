#自定义数据集
from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):
    def __init__(self,split):
        #从磁盘加载数据
        self.dataset = load_from_disk(r"G:\00AI-LM\05_基于 BERT 的中文评价情感分析2025年聚客大模型第三期\demo_5\data\ChnSentiCorp")
        if split == 'train':
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']

        return text,label

if __name__ == '__main__':
    dataset = MyDataset("test")
    for data in dataset:
        print(data)