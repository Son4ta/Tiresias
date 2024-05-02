import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# If you expect the results to be reproducible, set a random seed.
# torch.manual_seed(1234)
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained("model/Qwen-VL-Chat-Int4", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("model/Qwen-VL-Chat-Int4", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("model/Qwen-VL-Chat-Int4", trust_remote_code=True)
query = tokenizer.from_list_format([
    {'image': 'images/coconut.jpg'},
    {'text': 'It\'s April 25th, 2025. Is this still drinkable?Why?'},
])

process_time = time.time()

response, history = model.chat(tokenizer, query=query, history=None)
print(response)

end_time = time.time()
print("初始化耗时: {:.2f}秒".format(process_time - start_time))
print("推理耗时: {:.2f}秒".format(end_time - process_time))
