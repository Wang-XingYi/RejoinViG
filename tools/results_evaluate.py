"""
Calculate Top-K accuracy
"""

import csv
import pickle

from src.myutils import *
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
log = open(r'../logs/01_classify_log.txt', 'w', encoding='utf-8')


def calculate_top_k_accuracy(input_file, K_values=[1, 3, 5, 10, 15, 20], model_name="GreedViG"):
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=' ')
        for row in reader:
            row['pre_probability'] = float(row['pre_probability'])
            row['pre_classes'] = int(row['pre_classes'])
            row['label'] = int(row['label'])
            data.append(row)

    # sort
    grouped_data = {}
    for row in data:
        source_img = row['source_img']
        if source_img not in grouped_data:
            grouped_data[source_img] = []
        grouped_data[source_img].append(row)

    for source_img in grouped_data:
        grouped_data[source_img] = sorted(grouped_data[source_img], key=lambda x: -x['pre_probability'])

    # calculate Top-K accuracy
    total_source_images = len(grouped_data)
    top_k_correct = {k: 0 for k in K_values}
    for k in K_values:
        with open(f'../logs/{model_name}_top{k}.txt', 'w', encoding='utf-8') as topk_file:
            topk_file.write(f"source_img target_img pre_classes label\n")

    for source_img, targets in grouped_data.items():
        for k in K_values:
            valid_count = 0
            for target in targets:
                if target['pre_classes'] != 4:
                    valid_count += 1
                    if target['pre_classes'] == target['label']:
                        top_k_correct[k] += 1
                        break
                if valid_count == k:
                    break

            # save source_img and target_img in Top-k
            valid_count_top_k = 0
            with open(f'../logs/{model_name}_top{k}.txt', 'a', encoding='utf-8') as topk_file:
                for target in targets:
                    if valid_count_top_k == k:
                        break
                    if target['pre_classes'] != 4:
                        topk_file.write(f"{source_img} {target['target_img']} {target['pre_classes']} {target['label']}\n")
                        valid_count_top_k += 1



    # Top-K accuracy
    for k in K_values:
        accuracy = top_k_correct[k] / total_source_images * 100
        print(f"Top-{k} accuracy is {accuracy:.1f}%")
        log.write(f"Top-{k} accuracy is {accuracy:.1f}%\n")


def read_txt_file(input_file):

    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        next(file)
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

    trueLabel = []


    for item in data:
        source_img, target_img, pre_probability, pre_classes, label = item
        trueLabel.append(label)


    predictions_np = [pred.cpu().numpy() for pred in predictions]


    Y_pred = np.array(predictions_np)
    trueLabel = np.array(trueLabel)


    Y_pred_classes = np.argmax(Y_pred, axis=1)


    report = classification_report(trueLabel, Y_pred_classes,
                                   target_names=['top_bottom', 'bottom_top', 'left_right', 'right_left', 'not_rejoining'],
                                   labels=[0, 1, 2, 3, 4])
    print(report)
    log.write('--------------------RejoinViG-------------------------\n')
    log.write(f"accuracy is {np.mean(Y_pred_classes == trueLabel):.4f} \n")
    log.write(f'\n{report}\n')


    index_calculation(Y_pred, Y_pred_classes, trueLabel, log)
    log.close()

if __name__ == '__main__':
    torch.cuda.set_device(1)

    input_file = '../logs/01_synthesis_dataset_record_log.txt'
    model_name = 'RejoinViG'
    predictions = []
    with open('../logs/01_synthesis_pred_file.pkl', 'rb') as f:
        while True:
            try:
                pred = pickle.load(f)
                predictions.append(pred)
            except EOFError:
                break

    calculate_top_k_accuracy(input_file, model_name=model_name)
    data = read_txt_file(input_file)
    evaluate_model(data,predictions)
