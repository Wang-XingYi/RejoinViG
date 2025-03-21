import csv

log = open(r'./logs/01_TopK_log.txt', 'w', encoding='utf-8')
"""
calculate Top-K accuracy
"""
def calculate_top_k_accuracy(input_file, K_values=[1, 3, 5, 10, 15, 20], top_15_output_file="RejoinViG.txt"):
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=' ')
        for row in reader:
            row['pre_probability'] = float(row['pre_probability'])
            row['pre_classes'] = int(row['pre_classes'])
            row['label'] = int(row['label'])
            data.append(row)

    # Group by source_img and sort by pre_probability
    grouped_data = {}
    for row in data:
        source_img = row['source_img']
        if source_img not in grouped_data:
            grouped_data[source_img] = []
        grouped_data[source_img].append(row)

    for source_img in grouped_data:
        grouped_data[source_img] = sorted(grouped_data[source_img], key=lambda x: -x['pre_probability'])

    # Calculate Top-K accuracy
    total_source_images = len(grouped_data)
    top_k_correct = {k: 0 for k in K_values}   # Record correct count for each K value

    # Save source_img and target_img for Top-15
    with open(top_15_output_file, 'w', encoding='utf-8') as top15_file:
        top15_file.write(f"source_img target_img pre_classes label\n")
        for source_img, targets in grouped_data.items():
            correct_found = False
            for k in K_values:

                valid_count = 0
                for target in targets:
                    # Exclude items with predicted class 4 (unrejoinable)
                    if target['pre_classes'] != 4:
                        valid_count += 1
                        # If the predicted class matches the true label, mark as correct prediction
                        if target['pre_classes'] == target['label']:
                            if correct_found==False:
                                target_img = target['target_img']
                                print(f'{source_img}--{target_img}')
                            top_k_correct[k] += 1
                            correct_found = True
                            break
                    if valid_count == k:
                        break

            # Save source_img and target_img in Top-15
            valid_count_top_15 = 0
            for target in targets:
                if valid_count_top_15 == 15:
                    break
                if target['pre_classes'] != 4:   # Exclude items with predicted class 4
                    top15_file.write(f"{source_img} {target['target_img']} {target['pre_classes']} {target['label']}\n")
                    valid_count_top_15 += 1


    for k in K_values:
        accuracy = top_k_correct[k] / total_source_images * 100
        print(f"Top-{k} accuracy: {accuracy:.1f}%")
        log.write(f"Top-{k} accuracy: {accuracy:.1f}%\n")



if __name__ == '__main__':
    # 示例调用
    input_file = './logs/01_synthesis_dataset_record_log.txt'
    top_15_output_file = './logs/RejoinViG.txt'

    calculate_top_k_accuracy(input_file, top_15_output_file=top_15_output_file)

