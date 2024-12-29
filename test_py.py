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


per_class_accuracy = {'Cassette deck': 100.0, 'Printer': 77.66666666666667, 'Medical equipment': 100.0, 'Computer mouse': 100.0, 'Scale': 66.66666666666666, 'Telephone': 66.66666666666666, 'Camera': 100.0, 'Ipod': 100.0, 'Remote control': 100.0, 'Billboard': 70.0, 'Surfboard': 100.0, 'Drink': 75.5, 'Bottle': 74.6551724137931, 'Wine': 77.77777777777779, 'Beer': 77.73333333333333, 'Boy': 66.64705882352942, 'Person': 66.61538461538458, 'Man': 64.68571428571428, 'Sports equipment': 65.28, 'Sports uniform': 74.59999999999998, 'Whiteboard': 42.857142857142854, 'Food': 70.47058823529412, 'Light switch': 100.0, 'Bicycle': 75.0, 'Bicycle wheel': 50.0, 'Vehicle registration plate': 71.42857142857143, 'Land vehicle': 83.33333333333334, 'Vehicle': 81.25, 'Wheel': 66.66666666666666, 'Building': 70.0, 'Convenience store': 69.23076923076923, 'Toy': 0.0, 'Woman': 71.35714285714286, 'Girl': 71.35714285714286, 'Clothing': 72.16666666666664, 'Footwear': 87.4375, 'Scoreboard': 85.71428571428571, 'Bookcase': 71.42857142857143, 'Book': 57.96296296296296, 'Furniture': 57.14285714285714, 'Shelf': 71.42857142857143, 'Laptop': 66.625, 'Computer monitor': 100.0, 'Sink': 80.0, 'Home appliance': 50.0, 'Refrigerator': 50.0, 'Table': 50.0, 'Coffee table': 50.0, 'Kitchen appliance': 72.16666666666667, 'Shorts': 74.75, 'Plant': 70.0, 'Tree': 50.0, 'Fashion accessory': 50.0, 'Tin can': 88.66666666666667, 'Poster': 75.53333333333333, 'Alarm clock': 25.0, 'Mobile phone': 66.66666666666666, 'Digital clock': 50.0, 'Watch': 40.0, 'Giraffe': 100.0, 'Horse': 100.0, 'Motorcycle': 100.0, 'Wall clock': 60.0, 'Football': 66.66666666666666, 'Ruler': 25.0, 'Barrel': 100.0, 'Bottle opener': 100.0, 'Perfume': 80.0, 'Coin': 61.0, 'Car': 88.88888888888889, 'Computer keyboard': 44.333333333333336, 'Calculator': 66.5, 'Office supplies': 69.9, 'Baseball glove': 50.0, 'Human face': 86.6, 'Ball': 33.33333333333333, 'Mammal': 72.16666666666667, 'Lantern': 33.0, 'Cocktail': 55.333333333333336, 'Picture frame': 50.0, 'Tableware': 100.0, 'Fast food': 58.25, 'Shower': 100.0, 'Mirror': 100.0, 'Plumbing fixture': 100.0, 'Bathroom accessory': 100.0, 'Pencil case': 100.0, 'Glove': 50.0, 'Drum': 100.0, 'Musical instrument': 100.0, 'Clock': 42.857142857142854, 'Candle': 50.0, 'Platter': 50.0, 'Waste container': 100.0, 'Tower': 100.0, 'Skyscraper': 100.0, 'Taxi': 100.0, 'Auto part': 100.0, 'Starfish': 66.5, 'Box': 66.5, 'Segway': 100.0, 'Bus': 100.0, 'Van': 100.0, 'Aircraft': 33.33333333333333, 'Bird': 49.5, 'Pancake': 49.5, 'Dessert': 49.5, 'Snack': 100.0, 'Television': 100.0, 'Human body': 66.5, 'Human hair': 66.5, 'Human head': 66.5, 'Stop sign': 100.0, 'Airplane': 100.0, 'Tent': 100.0, 'Shirt': 100.0, 'Trousers': 66.33333333333334, 'Kettle': 100.0, 'Bowl': 100.0, 'Countertop': 100.0, 'Pressure cooker': 100.0, 'Train': 50.0, 'Sunglasses': 49.5, 'Jacket': 49.5, 'Jeans': 49.5, 'Traffic sign': 50.0}
class_f1 = {'Cassette deck': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Printer': {'tp': 2.33, 'fp': 0.6699999999999999, 'tn': 0, 'fn': 0.6699999999999999}, 'Medical equipment': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Computer mouse': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Scale': {'tp': 2.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Telephone': {'tp': 2.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Camera': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Ipod': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Remote control': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Billboard': {'tp': 7.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Surfboard': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Drink': {'tp': 22.65, 'fp': 7.35, 'tn': 0, 'fn': 7.35}, 'Bottle': {'tp': 21.65, 'fp': 7.35, 'tn': 0, 'fn': 7.35}, 'Wine': {'tp': 7.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Beer': {'tp': 11.66, 'fp': 3.34, 'tn': 0, 'fn': 3.34}, 'Boy': {'tp': 11.33, 'fp': 5.67, 'tn': 0, 'fn': 5.67}, 'Person': {'tp': 34.639999999999986, 'fp': 17.36, 'tn': 0, 'fn': 17.36}, 'Man': {'tp': 22.639999999999997, 'fp': 12.36, 'tn': 0, 'fn': 12.36}, 'Sports equipment': {'tp': 16.32, 'fp': 8.68, 'tn': 0, 'fn': 8.68}, 'Sports uniform': {'tp': 18.649999999999995, 'fp': 6.35, 'tn': 0, 'fn': 6.35}, 'Whiteboard': {'tp': 3.0, 'fp': 4.0, 'tn': 0, 'fn': 4.0}, 'Food': {'tp': 11.98, 'fp': 5.02, 'tn': 0, 'fn': 5.02}, 'Light switch': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Bicycle': {'tp': 3.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Bicycle wheel': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Vehicle registration plate': {'tp': 5.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Land vehicle': {'tp': 5.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Vehicle': {'tp': 13.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Wheel': {'tp': 2.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Building': {'tp': 7.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Convenience store': {'tp': 9.0, 'fp': 4.0, 'tn': 0, 'fn': 4.0}, 'Toy': {'tp': 0.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Woman': {'tp': 9.99, 'fp': 4.01, 'tn': 0, 'fn': 4.01}, 'Girl': {'tp': 9.99, 'fp': 4.01, 'tn': 0, 'fn': 4.01}, 'Clothing': {'tp': 30.30999999999999, 'fp': 11.69, 'tn': 0, 'fn': 11.69}, 'Footwear': {'tp': 13.99, 'fp': 2.01, 'tn': 0, 'fn': 2.01}, 'Scoreboard': {'tp': 6.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Bookcase': {'tp': 5.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Book': {'tp': 15.65, 'fp': 11.35, 'tn': 0, 'fn': 11.35}, 'Furniture': {'tp': 4.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Shelf': {'tp': 5.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Laptop': {'tp': 5.33, 'fp': 2.67, 'tn': 0, 'fn': 2.67}, 'Computer monitor': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Sink': {'tp': 4.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Home appliance': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Refrigerator': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Table': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Coffee table': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Kitchen appliance': {'tp': 4.33, 'fp': 1.67, 'tn': 0, 'fn': 1.67}, 'Shorts': {'tp': 2.99, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Plant': {'tp': 7.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Tree': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Fashion accessory': {'tp': 2.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Tin can': {'tp': 2.66, 'fp': 0.33999999999999997, 'tn': 0, 'fn': 0.33999999999999997}, 'Poster': {'tp': 11.33, 'fp': 3.67, 'tn': 0, 'fn': 3.67}, 'Alarm clock': {'tp': 1.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Mobile phone': {'tp': 4.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Digital clock': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Watch': {'tp': 2.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Giraffe': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Horse': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Motorcycle': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Wall clock': {'tp': 3.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Football': {'tp': 2.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Ruler': {'tp': 1.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Barrel': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Bottle opener': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Perfume': {'tp': 4.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Coin': {'tp': 3.66, 'fp': 2.34, 'tn': 0, 'fn': 2.34}, 'Car': {'tp': 8.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Computer keyboard': {'tp': 2.66, 'fp': 3.34, 'tn': 0, 'fn': 3.34}, 'Calculator': {'tp': 2.66, 'fp': 1.3399999999999999, 'tn': 0, 'fn': 1.3399999999999999}, 'Office supplies': {'tp': 6.99, 'fp': 3.01, 'tn': 0, 'fn': 3.01}, 'Baseball glove': {'tp': 3.0, 'fp': 3.0, 'tn': 0, 'fn': 3.0}, 'Human face': {'tp': 4.33, 'fp': 0.6699999999999999, 'tn': 0, 'fn': 0.6699999999999999}, 'Ball': {'tp': 1.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Mammal': {'tp': 4.33, 'fp': 1.67, 'tn': 0, 'fn': 1.67}, 'Lantern': {'tp': 0.66, 'fp': 1.3399999999999999, 'tn': 0, 'fn': 1.3399999999999999}, 'Cocktail': {'tp': 1.6600000000000001, 'fp': 1.3399999999999999, 'tn': 0, 'fn': 1.3399999999999999}, 'Picture frame': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Tableware': {'tp': 4.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Fast food': {'tp': 2.33, 'fp': 1.67, 'tn': 0, 'fn': 1.67}, 'Shower': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Mirror': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Plumbing fixture': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Bathroom accessory': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Pencil case': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Glove': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Drum': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Musical instrument': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Clock': {'tp': 3.0, 'fp': 4.0, 'tn': 0, 'fn': 4.0}, 'Candle': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Platter': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Waste container': {'tp': 4.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Tower': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Skyscraper': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Taxi': {'tp': 4.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Auto part': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Starfish': {'tp': 1.33, 'fp': 0.6699999999999999, 'tn': 0, 'fn': 0.6699999999999999}, 'Box': {'tp': 1.33, 'fp': 0.6699999999999999, 'tn': 0, 'fn': 0.6699999999999999}, 'Segway': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Bus': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Van': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Aircraft': {'tp': 1.0, 'fp': 2.0, 'tn': 0, 'fn': 2.0}, 'Bird': {'tp': 0.99, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Pancake': {'tp': 0.99, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Dessert': {'tp': 0.99, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Snack': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Television': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Human body': {'tp': 1.33, 'fp': 0.6699999999999999, 'tn': 0, 'fn': 0.6699999999999999}, 'Human hair': {'tp': 1.33, 'fp': 0.6699999999999999, 'tn': 0, 'fn': 0.6699999999999999}, 'Human head': {'tp': 1.33, 'fp': 0.6699999999999999, 'tn': 0, 'fn': 0.6699999999999999}, 'Stop sign': {'tp': 2.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Airplane': {'tp': 3.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Tent': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Shirt': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Trousers': {'tp': 1.9900000000000002, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Kettle': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Bowl': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Countertop': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Pressure cooker': {'tp': 1.0, 'fp': 0.0, 'tn': 0, 'fn': 0.0}, 'Train': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}, 'Sunglasses': {'tp': 0.99, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Jacket': {'tp': 0.99, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Jeans': {'tp': 0.99, 'fp': 1.0099999999999998, 'tn': 0, 'fn': 1.0099999999999998}, 'Traffic sign': {'tp': 1.0, 'fp': 1.0, 'tn': 0, 'fn': 1.0}}

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
