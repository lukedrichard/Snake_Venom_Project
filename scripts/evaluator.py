import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, MulticlassAUROC, ConfusionMatrix
import json

def evaluate(device, model, data_loader, results_dir):
    output_dim = model.output_dim
    #confusion matrix, precision, recall, f1
    accuracy = Accuracy(task='multiclass', num_classes=output_dim, average='micro').to(device)
    precision = Precision(task='multiclass',num_classes=output_dim,average='macro').to(device)
    recall = Recall(task='multiclass',num_classes=output_dim,average='macro').to(device)
    f1 = F1Score(task='multiclass',num_classes=output_dim,average='macro').to(device)
    roc_auc = MulticlassAUROC(num_classes=output_dim)


    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch  # assuming labels are already numerical
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            probs = nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)

            _, preds = torch.max(outputs, 1)

            all_preds.append(preds)
            all_labels.append(labels)

    #get metrics

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    accuracy_score = accuracy(all_preds, all_labels).item() * 100
    precision_score = precision(all_preds, all_labels).item()
    recall_score = recall(all_preds, all_labels).item()
    f1_score = f1(all_preds, all_labels).item()
    roc_auc_score = roc_auc(all_probs, all_labels)


    print(f"Post-training evaluation:")
    print(f"Accuracy: {accuracy_score:.4f}")
    print(f"Precision: {precision_score:.4f}")
    print(f"Recall: {recall_score:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Multiclass ROC AUC: {roc_auc_score:.4f}")

    #save metrics in .json file
    metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
    "roc_auc": roc_auc_score.item()
    }

    # Save metrics to JSON file
    with open(results_dir + "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    #generate confusion matrix
    confmat = ConfusionMatrix(task="multiclass", num_classes=output_dim).to(device)
    cm = confmat(all_preds, all_labels)
    print(cm)

    #plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(results_dir + 'confusion_matrix.png')  # Save accuracy plot
    plt.close()  # Close the figure

    return
