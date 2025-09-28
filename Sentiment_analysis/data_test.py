from datasets import load_dataset,load_from_disk

#在线加载数据
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
# print(dataset)
#转存为CSV格式
# dataset.to_csv(path_or_buf=r"D:\PycharmProjects\disanqi\demo_5\data\hermes-function-calling-v1.csv")
#加载csv格式数据
# dataset = load_dataset(path="csv",data_files=r"D:\PycharmProjects\disanqi\demo_5\data\hermes-function-calling-v1.csv")
# print(dataset)
#加载缓存数据
dataset = load_from_disk(r"G:\00AI-LM\05_基于 BERT 的中文评价情感分析2025年聚客大模型第三期\demo_5\data\ChnSentiCorp")
print(dataset)

test_data = dataset["train"]
for data in test_data:
    print(data)