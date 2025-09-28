#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('bert-base-chinese\models--bert-base-chinese',cache_dir='/mnt/workspace/LLM')
