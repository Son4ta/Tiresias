import json
import csv
import re
import time
import numpy as np
import requests
import os
from pathlib import Path
import matplotlib
import torch
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import Tiresias as Tias
from tqdm import tqdm  # Import tqdm for the progress bar


def download_image(image_url, local_dir):
    # Check if the image already exists locally
    image_name = os.path.basename(image_url)
    local_path = os.path.join(local_dir, image_name)
    if os.path.exists(local_path):
        print(f"Image '{image_name}' already exists locally. Skipping download.")
        return local_path
    """Download an image from a URL and save it locally."""
    response = requests.get(image_url)
    if response.status_code == 200:
        image_name = os.path.basename(image_url)
        local_path = os.path.join(local_dir, image_name)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Image '{image_name}' downloaded and saved to '{local_path}'.")
        return local_path
    else:
        print(f"Failed to download image from {image_url}")
        return None


def get_model_answer(image_path, question, model):
    """Use the model to generate an answer based on the local image and question."""
    ocr_result = Tias.ocr(image_path)
    query = (
        question + " " + ocr_result + " Short answer."
    )
    process_time = time.time()
    _, output = model.exec(image_path, query)
    end_time = time.time()
    print("Inference Time: {:.2f} seconds".format(end_time - process_time))
    print(f"Inference Answer: {output}")
    return output


def evaluate_model_on_textvqa(json_file_path, model, output_csv_path, start_from=0, max_samples=None):
    """Evaluate the model on the TextVQA dataset and save results to a CSV file."""
    local_image_dir = './images'
    Path(local_image_dir).mkdir(parents=True, exist_ok=True)

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 总数 样本index 总分
    total_questions = 0
    index = 0
    sum_score = 0
    # Dictionary to track per class performance
    class_scores = {}
    class_stats = {}  # 统计每类的 TP, FP, TN, FN

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['question', 'local_image_path', 'human_answer', 'model_answer', 'score', 'object_class',
                      'img_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Use tqdm to show progress bar
        for item in tqdm(data['data'], desc="###TESTING Text_VQA###", total=index + max_samples):
            if index < start_from:
                index += 1
                continue
            # Stop processing if max sample limit is reached
            if max_samples is not None and total_questions >= max_samples:
                break

            image_url = item['flickr_original_url']
            local_image_path = download_image(image_url, local_image_dir)
            # Skip if the image could not be downloaded
            if local_image_path is None:
                continue

            question = item['question']
            human_answers = item['answers']
            object_classes = item['image_classes']

            model_answer = get_model_answer(local_image_path, question, model)
            score = calculate_score(model_answer, human_answers)
            sum_score += score

            # Update class performance tracking
            for object_class in object_classes:
                if object_class not in class_scores:
                    class_scores[object_class] = {'correct': 0, 'total': 0}
                    # 添加 F1 统计
                    class_stats[object_class] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
                class_scores[object_class]['total'] += 1
            for object_class in object_classes:
                class_scores[object_class]['correct'] += score / 100
                class_stats[object_class]['tp'] += score / 100
                # 每个错误答案都是一个错过的真正例
                class_stats[object_class]['fp'] += 1 - score / 100
                class_stats[object_class]['fn'] += 1 - score / 100

            writer.writerow({'question': question,
                             'local_image_path': local_image_path,
                             'human_answer': human_answers,
                             'model_answer': model_answer,
                             'score': score,
                             'img_url': image_url,
                             'object_class': object_class})
            total_questions += 1

    # Calculate overall and per-class accuracy
    accuracy = sum_score / total_questions
    class_accuracies = {cls: (scores['correct'] / scores['total']) * 100 for cls, scores in class_scores.items()}
    return accuracy, class_accuracies, class_stats


def calculate_score(model_answer, human_answers):
    """Calculate score based on word matches between model and human answers."""
    model_answer = re.sub(r'[^a-zA-Z0-9\s]', ' ', model_answer)
    model_words = set(model_answer.lower().split())
    match_count = 0
    for answer in human_answers:
        human_words = set(answer.lower().split())
        if model_words & human_words:
            match_count += 1
    if match_count >= 3:
        return 100
    elif match_count == 2:
        return 66
    elif match_count == 1:
        return 33
    return 0


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


# Example usage:
# Replace `your_model` with your actual model object.
model = Tias.Tiresias("llava")
overall_accuracy, per_class_accuracy, class_f1 = evaluate_model_on_textvqa('TextVQA_0.5.1_val.json',
                                                                           model,
                                                                           './results_0-100.csv',
                                                                           start_from=0,
                                                                           max_samples=100)
print(torch.cuda.memory_stats())
del model
print(torch.cuda.memory_stats())

print(f"Overall Accuracy: {overall_accuracy}%")
print("Per-Class Accuracy:")
for cls, acc in per_class_accuracy.items():
    print(f"{cls}: {acc:.2f}%")
print(per_class_accuracy)
print(class_f1)
plot_accuracy(per_class_accuracy)
plot_f1_scores(class_f1)
