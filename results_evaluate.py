import csv
import pickle

from myutils import matrixPlot, plotPictrue, auc1, plot_confusion_matrix, get_roc_auc, index_calculation
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
log = open(r'./logs/01_MobileNetV4_classify_log.txt', 'w', encoding='utf-8')
"""
计算top K准确率
"""
def calculate_top_k_accuracy(input_file, K_values=[1, 3, 5, 10, 15, 20], top_15_output_file="GreedViG.txt"):
    # 读取txt文件
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=' ')
        for row in reader:
            row['pre_probability'] = float(row['pre_probability'])  # 转换pre_probability为浮点数
            row['pre_classes'] = int(row['pre_classes'])  # 转换pre_classes为整数
            row['label'] = int(row['label'])  # 转换label为整数
            data.append(row)

    # 按source_img分组并按pre_probability排序
    grouped_data = {}
    for row in data:
        source_img = row['source_img']
        if source_img not in grouped_data:
            grouped_data[source_img] = []
        grouped_data[source_img].append(row)

    for source_img in grouped_data:
        grouped_data[source_img] = sorted(grouped_data[source_img], key=lambda x: -x['pre_probability'])

    # 计算Top-K准确率
    total_source_images = len(grouped_data)
    top_k_correct = {k: 0 for k in K_values}  # 用于记录各个K值的正确计数

    # 保存Top-15的source_img和target_img
    with open(top_15_output_file, 'w', encoding='utf-8') as top15_file:
        top15_file.write(f"source_img target_img pre_classes label\n")
        for source_img, targets in grouped_data.items():
            correct_found = False
            for k in K_values:

                valid_count = 0
                for target in targets:
                    # 排除预测类别为4的项
                    if target['pre_classes'] != 4:
                        valid_count += 1
                        # 如果预测类别和真实标签一致，标记为找到正确的预测
                        if target['pre_classes'] == target['label']:
                            if correct_found==False:
                                target_img = target['target_img']
                                print(f'{source_img}--{target_img}')
                            top_k_correct[k] += 1
                            correct_found = True
                            break  # 一旦找到正确的类别，退出循环
                    # 如果找到足够数量的Top-K，跳出循环
                    if valid_count == k:
                        break

            # 保存Top-15中的source_img和target_img
            valid_count_top_15 = 0
            for target in targets:
                if valid_count_top_15 == 15:
                    break
                if target['pre_classes'] != 4:  # 排除预测类别为4的项
                    top15_file.write(f"{source_img} {target['target_img']} {target['pre_classes']} {target['label']}\n")
                    valid_count_top_15 += 1

    # 计算并打印Top-K的准确率
    for k in K_values:
        accuracy = top_k_correct[k] / total_source_images * 100
        print(f"Top-{k} 准确率为: {accuracy:.1f}%")
        log.write(f"Top-{k} 准确率为: {accuracy:.1f}%\n")

def read_txt_file(input_file):
    # 读取txt文件内容
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        next(file)  # 跳过第一行的表头
        for line in file:
            parts = line.strip().split(' ')
            source_img = parts[0]
            target_img = parts[1]
            pre_probability = float(parts[2])
            pre_classes = int(parts[3])
            label = int(parts[4])
            data.append((source_img, target_img, pre_probability, pre_classes, label))
    return data

def evaluate_model(data, predictions):
    # 初始化数据结构
    trueLabel = []

    # 遍历数据并填充
    for item in data:
        source_img, target_img, pre_probability, pre_classes, label = item
        trueLabel.append(label)

    # 将每个torch tensor转换为numpy数组
    predictions_np = [pred.cpu().numpy() for pred in predictions]

    # 转换为NumPy数组
    Y_pred = np.array(predictions_np)
    trueLabel = np.array(trueLabel)

    # 计算并保存结果
    Y_pred_classes = np.argmax(Y_pred, axis=1)  # 按行找出最大值对应的类别
    matrixPlot(Y_pred_classes, trueLabel)
    fpr, tpr, roc_auc = auc1(trueLabel, Y_pred)
    plotPictrue(fpr, tpr, roc_auc)

    confusion_mtx = confusion_matrix(trueLabel, Y_pred_classes)
    plot_confusion_matrix(confusion_mtx, classes=range(5))


    report = classification_report(trueLabel, Y_pred_classes,
                                   target_names=['top_bottom', 'bottom_top', 'left_right', 'right_left', 'not_rejoining'],
                                   labels=[0, 1, 2, 3, 4])
    print(report)
    log.write('--------------------所提方法（MobileNet V4）-------------------------\n')
    log.write(f"准确率为：{np.mean(Y_pred_classes == trueLabel):.4f} \n")
    log.write(f'\n{report}\n')

    # 计算各种评估指标
    index_calculation(Y_pred, Y_pred_classes, trueLabel, log)

    # 对应的标签
    labels = ['top_bottom', 'bottom_top', 'left_right', 'right_left', 'not_rejoining']

    # 创建DataFrame并保存为CSV文件
    # df = pd.DataFrame(confusion_mtx, index=labels, columns=['col1', 'col2', 'col3', 'col4', 'col5'])
    # df.to_csv('./logs/01_MobileNetV4_confusion_matrix.csv', sep='\t', header=False)
    log.close()

if __name__ == '__main__':
    # 示例调用
    input_file = './logs/01_synthesis_dataset_record_log.txt'  # 替换为你的输入文件路径
    top_15_output_file = './logs/RejoinViG.txt'  # 替换为保存Top-15结果的文件路径
    predictions = []
    with open('./logs/01_synthesis_pred_file.pkl', 'rb') as f:
        while True:
            try:
                pred = pickle.load(f)
                predictions.append(pred)
            except EOFError:
                break

    calculate_top_k_accuracy(input_file, top_15_output_file=top_15_output_file)
    data = read_txt_file(input_file)
    # 评估模型
    evaluate_model(data,predictions)
