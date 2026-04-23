import torch
import torchmetrics


def eval_classification_loss(predictions, true_y, logger = None, device="cpu"):
    if logger == None: logger=print
    metric_dict = dict()
    N_classes = len(predictions[0])
    CE_loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
    NegLL_loss = lambda predictions, true_y: -torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')(predictions.log(), true_y)
    ECE_metric = torchmetrics.classification.MulticlassCalibrationError(num_classes=N_classes, n_bins=15).to(device)
    acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=N_classes).to(device)
    auroc_metric = torchmetrics.classification.AUROC(task="multiclass", num_classes=N_classes).to(device)
    metric_names = ["LogLikelihood", "Expected Calibration Error", "Accuracy", "AUROC"]
    for metric, metric_name in zip([NegLL_loss, ECE_metric, acc_metric, auroc_metric], metric_names):
        measurement = metric(predictions, true_y)
        logger(f"{metric_name}: {measurement}")
        metric_dict[metric_name] = measurement.item()
    return metric_dict