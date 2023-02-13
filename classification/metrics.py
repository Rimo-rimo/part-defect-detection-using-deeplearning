from sklearn.metrics import classification_report
import wandb



def classification_metrics(true_list, pred_list, label_to_class, log_wandb):
    report = classification_report(true_list, pred_list, output_dict = True)
    result = dict()

    for i in range(len(label_to_class)):
        result[label_to_class[i]+"_Precision"] = report[str(i)]["precision"]
        result[label_to_class[i]+"_F1"] = report[str(i)]["f1-score"]
        result[label_to_class[i]+"_Recall"] = report[str(i)]["recall"]

    result["ACC"] = report["accuracy"]
    result["Precision"] = report["macro avg"]["precision"]
    result["Recall"] = report["macro avg"]["recall"]
    result["F1-Score"] = report["macro avg"]["f1-score"]
    if log_wandb:
        wandb.log(result)

    return result
