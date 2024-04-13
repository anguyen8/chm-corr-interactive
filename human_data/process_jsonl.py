import tarfile
import json
from collections import defaultdict
import numpy as np


from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

parent_dir = '/home/giang/Downloads'
train_data = ImageFolder(
    # ImageNet train folder
    root=f"/home/giang/Downloads/datasets/CUB/train1", transform=data_transform
)

class_to_idx = train_data.class_to_idx
name_to_idx = {}
for k, v in class_to_idx.items():
    part1, part2 = k.split('.')
    name_to_idx[part2.lower()] = part1

# Open the tar.gz file
tar_path = 'jsonl.tar.gz'  # Make sure to use your actual tar.gz file path
tar = tarfile.open(tar_path, "r:gz")

# Assuming you have a function to process and load interactions
def load_positive_interactions(tar, base_name):
    interactions_path = f'{base_name}_interactions.jsonl'
    interactions_data = {}
    f = tar.extractfile(interactions_path)
    if f:
        for line in f:
            data = json.loads(line)
            # Simplified: assuming each line in _interactions.jsonl has an image-index and chm_predictions_top_1
            image_index = data.get('image-index') or data.get('image_index')
            top_1_label = (next(iter(data['chm_predictions'].keys()))).split(' ')[0]  # Simplified extraction
            if image_index not in interactions_data:
                interactions_data[image_index] = [top_1_label]
            else:
                interactions_data[image_index].append(top_1_label)

    consitent_dict = {}
    top1_dict = {}
    for k, v in interactions_data.items():
        num_class = len(set(v))
        if num_class == 1:
            consistent = True
            top1_dict[k] = v[0]
        else:
            consistent = False
        consitent_dict[k] = consistent

    return consitent_dict, top1_dict


# Assuming you have a function to process and load interactions
def load_negative_interactions(tar, base_name):
    interactions_path = f'{base_name}_interactions.jsonl'
    interactions_data = {}
    f = tar.extractfile(interactions_path)
    if f:
        for line in f:
            data = json.loads(line)
            # Simplified: assuming each line in _interactions.jsonl has an image-index and chm_predictions_top_1
            image_index = data.get('image-index') or data.get('image_index')
            top_1_label = (next(iter(data['chm_predictions'].keys()))).split(' ')[0]  # Simplified extraction
            if image_index not in interactions_data:
                interactions_data[image_index] = [top_1_label]
            else:
                interactions_data[image_index].append(top_1_label)

    return interactions_data

def is_bad_jsonl(fileobj):
    count = 0
    for line in fileobj:
        try:
            json.loads(line)  # Try to parse the JSON line
            count += 1
            if count > 20:  # Return False only when count is exactly 20
                return True  # Considered bad if the file has more than 20 entries
        except json.JSONDecodeError:
            return True  # Bad file if there's an error in parsing
    return count != 20  # Return False if the count is exactly 20, True otherwise

def determine_method(interactions_name, decisions_name):
    return "CHMCorr++" if interactions_name and decisions_name else "CHMCorr"

method_metrics = defaultdict(lambda: defaultdict(list))


correct_consistent_decisions = 0
correct_inconsistent_decisions = 0
total_consistent_decisions = 0
total_inconsistent_decisions = 0

total_non_correction_decisions = 0
correct_non_correction_decisions = 0
total_correction_decisions = 0
correct_correction_decisions = 0

for member in tar.getmembers():
    if member.name.endswith('_decisions.jsonl'):
        f = tar.extractfile(member)
        if f and not is_bad_jsonl(f):  # Remove bad users
            base_name = member.name.replace('_decisions.jsonl', '')
            method = determine_method(f'{base_name}_interactions.jsonl' in tar.getnames(), True)

            TP, TN, FP, FN = 0, 0, 0, 0
            correct_human_decisions_correct_AI, correct_human_decisions_wrong_AI = 0, 0
            correct_AI_count, wrong_AI_count = 0, 0

            # When AI is correct
            if method == 'CHMCorr++':
                consistent_dict, top1_dict = load_positive_interactions(tar, base_name)
                for i in range(20):
                    if i not in consistent_dict:  # fill in the missing index
                        consistent_dict[i] = True

            # When AI is wrong
            if method == 'CHMCorr++':
                interactions_data = load_negative_interactions(tar, base_name)
                for i in range(20):
                    if i not in interactions_data:  # fill in the missing index
                        interactions_data[i] = []

            f.seek(0)  # Reset file pointer to the beginning
            for line in f:
                entry = json.loads(line.strip())
                gt_label = entry.get('gt-label', '').lower().replace('_', '').replace('-', '')
                ai_prediction_label = entry.get('ai-prediction', {}).get('label', '').split(' ')[-1].lower().replace('_', '').replace('-', '')
                decision_made = entry.get('decision', '')
                expected_decision = 'Accept' if gt_label == ai_prediction_label else 'Reject'

                # Examine if the AI is consistent about its prediction on this sample
                # Process only for CHMCorr++ and when AI prediction matches GT
                if method == 'CHMCorr++' and expected_decision == 'Accept':
                    img_index = entry.get('image-index')
                    consistency = consistent_dict[img_index]

                    gt_id = entry.get('ai-prediction')['label']
                    gt_id = gt_id.split(' ')[0]

                    # If img_index not in top1_dict, it means there is no further interaction,
                    # CHMCorr == CHMCorr++, AI is consistent
                    if img_index not in top1_dict:
                        total_consistent_decisions += 1
                        if decision_made == 'Accept':
                            correct_consistent_decisions += 1
                    # If human interacts, and AI always keep the same top-1 label as the original one -->
                    # AI is consistent
                    elif consistency is True and top1_dict[img_index] == gt_id:
                        total_consistent_decisions += 1
                        if decision_made == 'Accept':
                            correct_consistent_decisions += 1
                    else:  # AI is not consistent because it changes the top-1 label after human interaction
                        total_inconsistent_decisions += 1
                        if decision_made == 'Accept':
                            correct_inconsistent_decisions += 1

                ##########################

                if method == 'CHMCorr++' and expected_decision == 'Reject':
                    img_index = entry.get('image-index')
                    seen_labels = interactions_data[img_index]

                    gt = entry.get('gt-label')
                    gt = gt.split(' ')[0].lower()

                    gt_id = name_to_idx[gt]

                    if len(seen_labels) == 0:  #If humans never interact
                        total_non_correction_decisions += 1
                        if decision_made == 'Reject':
                            correct_non_correction_decisions += 1
                    elif gt_id in seen_labels:  # if humans ever make the AI correctly classify via interaction
                        total_correction_decisions += 1
                        if decision_made == 'Reject':
                            correct_correction_decisions += 1
                    else:  # humans never can help AI classify correctly via interaction
                        total_non_correction_decisions += 1
                        if decision_made == 'Reject':
                            correct_non_correction_decisions += 1

                ##########################

                if decision_made == 'Accept' and expected_decision == 'Accept':
                    correct_human_decisions_correct_AI += 1
                    TP += 1
                elif decision_made == 'Reject' and expected_decision == 'Reject':
                    correct_human_decisions_wrong_AI += 1
                    TN += 1
                elif decision_made == 'Accept' and expected_decision == 'Reject':
                    FP += 1
                elif decision_made == 'Reject' and expected_decision == 'Accept':
                    FN += 1

                if expected_decision == 'Accept':
                    correct_AI_count += 1
                else:
                    wrong_AI_count += 1

            # At the end of your existing for loop where you calculate TP, TN, FP, FN
            # Add the TP, TN, FP, FN to the method_metrics
            method_metrics[method]['TP'].append(TP)
            method_metrics[method]['TN'].append(TN)
            method_metrics[method]['FP'].append(FP)
            method_metrics[method]['FN'].append(FN)

            total_samples = TP + FP + FN + TN
            correct_decisions = TP + TN
            accuracy = (correct_decisions / total_samples) * 100 if total_samples > 0 else 0
            correct_AI_accuracy = (correct_human_decisions_correct_AI / correct_AI_count) * 100 if correct_AI_count > 0 else 0
            wrong_AI_accuracy = (correct_human_decisions_wrong_AI / wrong_AI_count) * 100 if wrong_AI_count > 0 else 0

            # Store metrics
            method_metrics[method]['accuracies'].append(accuracy)
            method_metrics[method]['correct_AI_accuracies'].append(correct_AI_accuracy)
            method_metrics[method]['wrong_AI_accuracies'].append(wrong_AI_accuracy)
            method_metrics[method]['correct_human_decisions_correct_AI'].append(correct_human_decisions_correct_AI)
            method_metrics[method]['correct_human_decisions_wrong_AI'].append(correct_human_decisions_wrong_AI)
            method_metrics[method]['correct_AI_count'].append(correct_AI_count)
            method_metrics[method]['wrong_AI_count'].append(wrong_AI_count)

tar.close()

method_stats = {}
for method, metrics in method_metrics.items():
    total_correct_human_decisions_correct_AI = sum(metrics['correct_human_decisions_correct_AI'])
    total_correct_human_decisions_wrong_AI = sum(metrics['correct_human_decisions_wrong_AI'])
    total_correct_AI_count = sum(metrics['correct_AI_count'])
    total_wrong_AI_count = sum(metrics['wrong_AI_count'])

    method_stats[method] = {
        'mean_accuracy': np.mean(metrics['accuracies']),
        'std_accuracy': np.std(metrics['accuracies']),
        'mean_correct_AI_accuracy': np.mean(metrics['correct_AI_accuracies']),
        'std_correct_AI_accuracy': np.std(metrics['correct_AI_accuracies']),
        'mean_wrong_AI_accuracy': np.mean(metrics['wrong_AI_accuracies']),
        'std_wrong_AI_accuracy': np.std(metrics['wrong_AI_accuracies']),
        'num_trials': len(metrics['accuracies']),
        'total_correct_human_decisions_correct_AI': total_correct_human_decisions_correct_AI,
        'total_correct_human_decisions_wrong_AI': total_correct_human_decisions_wrong_AI,
        'total_correct_AI_count': total_correct_AI_count,
        'total_wrong_AI_count': total_wrong_AI_count,
    }

# Calculate accuracies
accuracy_consistent = (correct_consistent_decisions / total_consistent_decisions * 100) if total_consistent_decisions > 0 else 0
accuracy_inconsistent = (correct_inconsistent_decisions / total_inconsistent_decisions * 100) if total_inconsistent_decisions > 0 else 0

print(correct_consistent_decisions, total_consistent_decisions)
print(f"CHM-Corr++ Accuracy when AI is correct and consistent: {accuracy_consistent}%")
print(correct_inconsistent_decisions, total_inconsistent_decisions)
print(f"CHM-Corr++ Accuracy when AI is correct and inconsistent: {accuracy_inconsistent}%")


accuracy_negative_non_correction = (correct_non_correction_decisions / total_non_correction_decisions * 100) if total_non_correction_decisions > 0 else 0
accuracy_negative_correction = (correct_correction_decisions / total_correction_decisions * 100) if total_correction_decisions > 0 else 0

print(correct_non_correction_decisions, total_non_correction_decisions)
print(f"CHM-Corr++ Accuracy when AI is wrong and never classify correctly: {accuracy_negative_non_correction}%")
print(correct_correction_decisions, total_correction_decisions)
print(f"CHM-Corr++ Accuracy when AI is wrong and EVER classify correctly: {accuracy_negative_correction}%")


for key, val in method_stats.items():
    print(f"Method: {key}")
    for stat_key, stat_val in val.items():
        print(f"{stat_key}: {stat_val}")
    print()  # Add an empty line for better readability between methods

# breakpoint()

from scipy.stats import ttest_ind

CHMCorr_accuracies =  method_metrics['CHMCorr']['accuracies']
CHMCorrPlusPlus_accuracies =  method_metrics['CHMCorr++']['accuracies']

# Perform the t-test on these accuracies
t_stat, p_value = ttest_ind(CHMCorr_accuracies, CHMCorrPlusPlus_accuracies)

print('t_stat:', t_stat)
print('p_value:', p_value   )


# After processing all files and calculating metrics, before closing the tar file
# Calculate and print the confusion matrix for each method
for method, metrics in method_metrics.items():
    # Sum up TP, TN, FP, FN across all files for the current method
    total_TP = sum(metrics['TP'])
    total_TN = sum(metrics['TN'])
    total_FP = sum(metrics['FP'])
    total_FN = sum(metrics['FN'])

    # Construct the confusion matrix
    confusion_matrix = np.array([[total_TP, total_FN], [total_FP, total_TN]])

    print(f"Method: {method}")
    print("Confusion Matrix:")
    print(confusion_matrix)
    print()  # Add an empty line for better readability between methods


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Assuming the rest of your script is unchanged and just before tar.close()

def plot_confusion_matrix(cm, method_name, labels=['Accept', 'Reject']):
    """
    Plots a confusion matrix.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    ax.figure.colorbar(im, ax=ax)

    if method_name == 'CHMCorr++':
        title_name = 'CHM-Corr++'
    else:
        title_name = 'CHM-Corr'

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=f'Confusion Matrix for {title_name}',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


for method, metrics in method_metrics.items():
    total_TP = sum(metrics['TP'])
    total_TN = sum(metrics['TN'])
    total_FP = sum(metrics['FP'])
    total_FN = sum(metrics['FN'])

    # Construct the confusion matrix
    confusion_matrix = np.array([[total_TP, total_FN], [total_FP, total_TN]])

    # Plot and save the confusion matrix
    plot_confusion_matrix(confusion_matrix, method)
    plt.savefig(f'confusion_matrix_{method}.pdf',  bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the plot to free up memory

