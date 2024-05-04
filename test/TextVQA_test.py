import json
import csv
import time

import numpy as np
import requests
import os
from pathlib import Path
import Tiresias as Tias


def download_image(image_url, local_dir):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_name = os.path.basename(image_url)
        local_path = os.path.join(local_dir, image_name)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        return local_path
    return None


def get_model_answer(local_image_path, question, model):
    ocr_result = Tias.ocr(local_image_path)
    query = (  # speech_recognition(audio_file)
        question
        + ocr_result
        + "Short answer."
    )
    process_time = time.time()
    _, output = model.exec(local_image_path, query)
    end_time = time.time()
    print("推理耗时: {:.2f}秒".format(end_time - process_time))
    print(f"推理回答: {output}")
    return output


def evaluate_model_on_textvqa(json_file_path, model, output_csv_path, max_samples=None):
    local_image_dir = './images'
    Path(local_image_dir).mkdir(parents=True, exist_ok=True)

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    total_questions = 0

    # Dictionary to track per class performance
    class_scores = {}

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['question', 'local_image_path', 'human_answer', 'model_answer', 'score', 'object_class', 'img_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in data['data']:
            if max_samples is not None and total_questions >= max_samples:
                break  # Stop processing if max sample limit is reached

            image_url = item['flickr_original_url']
            local_image_path = download_image(image_url, local_image_dir)
            if local_image_path is None:
                continue  # Skip if the image could not be downloaded

            question = item['question']
            human_answers = item['answers']
            object_classes\

            model_answer = get_model_answer(local_image_path, question, model)

            score = calculate_score(model_answer, human_answers)

            # Update class performance tracking
            for object_class in object_classes:
                if object_class not in class_scores:
                    class_scores[object_class] = {'correct': 0, 'total': 0}
                class_scores[object_class]['total'] += 1
            if score == 100:
                class_scores[object_class]['correct'] += 1

            writer.writerow({'question': question,
                             'local_image_path': local_image_path,
                             'human_answer': human_answers,
                             'model_answer': model_answer,
                             'score': score,
                             'img_url': image_url,
                             'object_class': object_class})
            total_questions += 1

    # Calculate overall and per-class accuracy
    accuracy = calculate_overall_accuracy(class_scores, total_questions)
    class_accuracies = {cls: (scores['correct'] / scores['total']) * 100 for cls, scores in class_scores.items()}
    return accuracy, class_accuracies


def calculate_score(model_answer, human_answers):
    match_count = sum(1 for ans in human_answers if ans.lower() == model_answer.lower())
    if match_count >= 3:
        return 100
    elif match_count == 2:
        return 66
    elif match_count == 1:
        return 33
    return 0


def calculate_overall_accuracy(class_scores, total_questions):
    correct_answers = sum(scores['correct'] for scores in class_scores.values())
    return (correct_answers / total_questions) * 100


# Example usage:
# Replace `your_model` with your actual model object.
model = Tias.Tiresias("llava")
overall_accuracy, per_class_accuracy = evaluate_model_on_textvqa('TextVQA_0.5.1_val.json', model, './results.csv',
                                                                 max_samples=10)
print(f"Overall Accuracy: {overall_accuracy}%")
print("Per-Class Accuracy:")
for cls, acc in per_class_accuracy.items():
    print(f"{cls}: {acc:.2f}%")
