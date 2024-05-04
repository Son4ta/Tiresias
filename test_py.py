import os
import time
import torch
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


def calculate_f1_scores(class_stats):
    f1_scores = {}
    for cls, stats in class_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[cls] = f1
    return f1_scores


def plot_accuracy(class_accuracies):
    classes = list(class_accuracies.keys())
    accuracies = [acc for acc in class_accuracies.values()]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

    plt.figure(figsize=(10, 5))
    plt.bar(classes, accuracies, color='skyblue')
    plt.axhline(y=avg_accuracy, color='r', linestyle='--')
    plt.text(0.95, avg_accuracy, f'avg_acc: {avg_accuracy:.2f}', va='bottom', ha='right', color='red')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per Class Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_f1_scores(class_stats):
    f1_scores = calculate_f1_scores(class_stats)
    classes = list(f1_scores.keys())
    f1_values = list(f1_scores.values())

    plt.figure(figsize=(10, 5))
    plt.plot(classes, f1_values, marker='o', linestyle='-', color='green')
    plt.xlabel('类别')
    plt.ylabel('F1 分数')
    plt.title('每个类别的 F1 分数曲线')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.pause(0.01)  # plt.ion() 留出时间让图像显示，解决不显示图像问题
    plt.show(block=True)  # 避免plt与其他包冲突，解决未响应问题


per_class_accuracy ={'Football': 100.0, 'Person': 100.0, 'Plant': 100.0, 'Sports equipment': 100.0}
class_f1 = {'Football': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Person': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Plant': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Sports equipment': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}}
plot_accuracy(per_class_accuracy)
plot_f1_scores(class_f1)
# os.system("shutdown -s -t  60 ")
# # If you expect the results to be reproducible, set a random seed.
# # torch.manual_seed(1234)
# start_time = time.time()
#
# tokenizer = AutoTokenizer.from_pretrained("model/Qwen-VL-Chat-Int4", trust_remote_code=True)
#
# model = AutoModelForCausalLM.from_pretrained("model/Qwen-VL-Chat-Int4", device_map="cuda", trust_remote_code=True).eval()
# model.generation_config = GenerationConfig.from_pretrained("model/Qwen-VL-Chat-Int4", trust_remote_code=True)
# query = tokenizer.from_list_format([
#     {'image': 'images/coconut.jpg'},
#     {'text': 'It\'s April 25th, 2025. Is this still drinkable?Why?'},
# ])
#
# process_time = time.time()
#
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
#
# end_time = time.time()
# print("初始化耗时: {:.2f}秒".format(process_time - start_time))
# print("推理耗时: {:.2f}秒".format(end_time - process_time))
