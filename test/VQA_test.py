import csv
import json
import os
import re
import time
from pathlib import Path

import matplotlib
import torch
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
import Tiresias as Tias
from tqdm import tqdm  # Import tqdm for the progress bar


def load_image(image_id, local_dir):
    """Download an image from a URL and save it locally."""
    # Check if the image already exists locally
    local_path = os.path.join(local_dir, image_id + ".jpg")
    if os.path.exists(local_path):
        print(f"【Load SUCCESS】>>> Image '{image_id}'.jpg exists locally. LOADING.")
        return local_path
    else:
        print(f"【Load FAILED】>>> Failed to load image from {local_path}")
        return None


def get_model_answer(image_path, question, model, ocr=True):
    """Use the model to generate an answer based on the local image and question."""
    if ocr:
        ocr_result, target_num, avg_conf = Tias.ocr(image_path)
        query = (
            question + " " + ocr_result + " Short answer."
        )
        process_time = time.time()
        _, output = model.exec(image_path, query)
        end_time = time.time()
        print("【LLM】>>> Inference Time: {:.2f} seconds".format(end_time - process_time))
        print(f"【LLM】>>> Inference Answer: {output}")
        return output, ocr_result, target_num, avg_conf, end_time - process_time
    else:
        query = (
            question + " Short answer."
        )
        process_time = time.time()
        _, output = model.exec(image_path, query)
        end_time = time.time()
        print("【LLM】>>> Inference Time: {:.2f} seconds".format(end_time - process_time))
        print(f"【LLM】>>> Inference Answer: {output}")
        return output, "N/A", "N/A", "N/A", end_time - process_time


def evaluate_model_on_textvqa(json_file_path, model, output_csv_path, start_from=0, max_samples=None, ocr=True):
    """Evaluate the model on the TextVQA dataset and save results to a CSV file."""
    # 图片保存目录
    local_image_dir = './images/train_images'
    Path(local_image_dir).mkdir(parents=True, exist_ok=True)
    # 打开json
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 总数 样本index 总分
    total_questions = 0
    index = 0
    sum_score = 0
    # Dictionary to track per class performance
    class_scores = {}

    with open(output_csv_path, 'w', newline='') as csvfile:
        # csv列名
        fieldnames = ['index',
                      'question',
                      'local_image_path',
                      'human_answer',
                      'model_answer',
                      'ocr_result',
                      'target_num',
                      'ocr_conf',
                      'score',
                      'object_class',
                      'img_id',
                      'inference_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Use tqdm to show progress bar
        for item in tqdm(data['data'], desc="###TESTING Text_VQA###", total=start_from + max_samples):
            if index < start_from:
                index += 1
                continue
            # Stop processing if max sample limit is reached
            if max_samples is not None and total_questions >= max_samples:
                break
            print(f"【Testing】>>> INDEX:  [{index}]")
            # 读取图片链接
            image_id = item['image_id']
            local_image_path = load_image(image_id, local_image_dir)
            # Skip if the image could not be downloaded
            if local_image_path is None:
                index += 1
                continue
            # 读取json
            question = item['question']
            human_answers = item['answers']
            object_classes = item['image_classes']
            # 处理掉一些奇怪的字符免得csv写入出错
            for i in range(len(human_answers)):
                human_answers[i] = re.sub(r'[^a-zA-Z0-9\s]', '', human_answers[i])
            # 推理并评分
            model_answer, ocr_result, target_num, ocr_conf, inference_time = get_model_answer(local_image_path,
                                                                                              question,
                                                                                              model,
                                                                                              ocr=ocr)
            score = calculate_score(model_answer, human_answers)
            print(f"【Score】:{score}")
            sum_score += score

            # Update class performance tracking
            for object_class in object_classes:
                if object_class not in class_scores:
                    class_scores[object_class] = {'correct': 0, 'total': 0}
                class_scores[object_class]['total'] += 1
            for object_class in object_classes:
                class_scores[object_class]['correct'] += score / 100
            # 写入表格记录数据并捕获可能的异常
            try:
                writer.writerow({'index': index,
                                 'question': question,
                                 'local_image_path': local_image_path,
                                 'human_answer': human_answers,
                                 'model_answer': model_answer,
                                 'ocr_result': ocr_result,
                                 'target_num': target_num,
                                 'ocr_conf': ocr_conf,
                                 'score': score,
                                 'img_id': image_id,
                                 'object_class': object_class,
                                 'inference_time': inference_time})
            except UnicodeEncodeError as e:
                print(f"【Error】Unicode error encountered at index {index}: {e}")
                writer.writerow({'index': index})
                index += 1
                continue
            except Exception as e:
                print(f"【Error】Unexpected error at index {index}: {e}")
                writer.writerow({'index': index})
                index += 1
                continue
            # 计数
            index += 1
            total_questions += 1

    # Calculate overall and per-class accuracy
    accuracy = sum_score / total_questions
    class_accuracies = {cls: (scores['correct'] / scores['total']) * 100 for cls, scores in class_scores.items()}
    return accuracy, class_accuracies


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


model = Tias.Tiresias("llava")
overall_accuracy, per_class_accuracy = evaluate_model_on_textvqa('TextVQA_0.5.1_val.json',
                                                                 model,
                                                                 './699-2400.csv',
                                                                 start_from=699,
                                                                 max_samples=1701,
                                                                 ocr=True)
del model

print(f"Overall Accuracy: {overall_accuracy}%")
print("Per-Class Accuracy:")
for cls, acc in per_class_accuracy.items():
    print(f"{cls}: {acc:.2f}%")
print(per_class_accuracy)
plot_accuracy(per_class_accuracy)
# os.system("shutdown -s -t  60 ")
