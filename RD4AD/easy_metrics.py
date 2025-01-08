import numpy as np

import wandb


def wandb_metric_logs(class_name, all_images, total_pred, total_label):
    from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve

    preds = np.concatenate((total_pred.reshape((-1, 1)), total_label.reshape((-1, 1)), np.array(all_images).reshape((-1, 1))), axis=1)

    binary_pred = np.concatenate(((1 - total_pred).reshape((-1, 1)), total_pred.reshape((-1, 1))), axis=1)

    wandb.log({f'{class_name}_pred': wandb.Table(data=preds, columns=['total_pred', 'total_label', 'image'])})
    wandb.log({f'{class_name}_ROC': wandb.plot.roc_curve(total_label, binary_pred, labels=[f'{class_name}_normal', f'{class_name}_defect'])})
    wandb.log({f'{class_name}_PR': wandb.plot.pr_curve(total_label, binary_pred, labels=[f'{class_name}_normal', f'{class_name}_defect'])})

    #roc_auc, ap = aucPerformance(total_pred, total_label, prt=False)

    roc_auc = roc_auc_score(total_label, total_pred)
    ap = average_precision_score(total_label, total_pred)

    fpr, tpr, threshold = roc_curve(total_label, total_pred)
    precision, recall, pr_threshold = precision_recall_curve(total_label, total_pred)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    opt_pr_thresh = pr_threshold[np.argmax(f1_scores)]
    max_f1_score = np.max(f1_scores)

    opt_thresh = threshold[np.argmax(tpr - fpr)]
    concrete_pred = (total_pred > opt_thresh).astype(float)

    opt_acc = accuracy_score(total_label, concrete_pred)
    opt_precision = precision_score(total_label, concrete_pred)
    opt_recall = recall_score(total_label, concrete_pred)
    opt_f1_score = f1_score(total_label, concrete_pred)

    fpr_table = wandb.Table(data=[[x, y] for (x, y) in zip(threshold, fpr)], columns=['threshold', 'fpr'])
    fpr_line = wandb.plot.line(fpr_table, 'threshold', 'fpr', title=f'{class_name} False positive rate curve')

    tpr_table = wandb.Table(data=[[x, y] for (x, y) in zip(threshold, tpr)], columns=['threshold', 'tpr'])
    tpr_line = wandb.plot.line(tpr_table, 'threshold', 'tpr', title=f'{class_name} True positive rate curve')

    precision_table = wandb.Table(data=[[x, y] for (x, y) in zip(pr_threshold, precision)], columns=['threshold', 'precision'])
    precision_line = wandb.plot.line(precision_table, 'threshold', 'precision', title=f'{class_name} Precisison curve')

    recall_table = wandb.Table(data=[[x, y] for (x, y) in zip(pr_threshold, recall)], columns=['threshold', 'recall'])
    recall_line = wandb.plot.line(recall_table, 'threshold', 'recall', title=f'{class_name} Recall curve')

    wandb.log({
        f'{class_name}_fpr': fpr_line,
        f'{class_name}_tpr': tpr_line,
        f'{class_name}_precision': precision_line,
        f'{class_name}_recall': recall_line,
        f'{class_name}_ROC_AUC': roc_auc,
        f'{class_name}_AUC_PR': ap,
        f'{class_name}_opt_thresh': opt_thresh,
        f'{class_name}_opt_acc': opt_acc,
        f'{class_name}_opt_precision': opt_precision,
        f'{class_name}_opt_recall': opt_recall,
        f'{class_name}_opt_f1_score': opt_f1_score,
        f'{class_name}_opt_pr_thresh': opt_pr_thresh,
        f'{class_name}_max_f1_score': max_f1_score,
    })
