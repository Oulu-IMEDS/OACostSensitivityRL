from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, mean_absolute_error, roc_auc_score, recall_score, precision_score


def to_cpu(x: torch.Tensor or torch.cuda.FloatTensor, required_grad=False, use_numpy=True):
    x_cpu = x

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            if use_numpy:
                x_cpu = x.detach().cpu().numpy()
            elif required_grad:
                x_cpu = x.cpu()
            else:
                x_cpu = x.cpu().required_grad_(False)
        elif use_numpy:
            x_cpu = x.numpy()

    return x_cpu


def collect_and_update_metrics(metrics_collector, new_record):
    # Collect data
    metrics_collector['loss'].append(new_record['new_loss'])
    np_preds_classes = to_cpu(torch.argmax(new_record['new_preds'], dim=1))
    np_labels = to_cpu(new_record['new_labels'])
    metrics_collector['pred'] += np_preds_classes.tolist()
    metrics_collector['label'] += np_labels.tolist()

    # Calculate metrics
    _loss = np.mean(np.array(metrics_collector['loss']))
    _ba = balanced_accuracy_score(metrics_collector['label'], metrics_collector['pred'])
    _mae = mean_absolute_error(metrics_collector['label'], metrics_collector['pred'])
    _kappa_quad = cohen_kappa_score(metrics_collector['label'], metrics_collector['pred'], weights="quadratic")

    # Put metrics into the dictionary metrics for display in progress bar
    metrics_display = OrderedDict()
    metrics_display['loss'] = _loss
    metrics_display['balanced_accuracy'] = _ba
    # metrics_display['mean_abs_err'] = _mae
    # metrics_display['kappa_quad'] = _kappa_quad
    return metrics_collector, metrics_display

def calculate_metrics_by_preds(cfg, metrics_collector, metric_name='balanced-accuracy'):
    if metric_name == 'balanced-accuracy':
        score = balanced_accuracy_score(metrics_collector['label'], metrics_collector['pred'])
    elif metric_name == 'mae':
        score = mean_absolute_error(metrics_collector['label'], metrics_collector['pred'])
    elif metric_name == 'cost':
        score, _ = convert_predictions_to_cost(cfg, metrics_collector['label'], metrics_collector['pred'])
    elif metric_name == 'recall':
        score = recall_score(metrics_collector['label'], metrics_collector['pred'])
    elif metric_name == 'precision':
        score = precision_score(metrics_collector['label'], metrics_collector['pred'])
    elif metric_name == 'roc-auc':
        if len(np.unique(metrics_collector['label'])) > 2:
            score = roc_auc_score(metrics_collector['label'], metrics_collector['probs'], average='macro',
                                                                        labels=[i for i in range(cfg.time_interval + 1)],
                                                                        multi_class='ovo')
        else:
            score = roc_auc_score(metrics_collector['label'], metrics_collector['probs'])
    else:
        raise ValueError('Not support this metric')

    return score

def convert_predictions_to_cost(cfg, label, pred):
    n_steps = len(np.unique(label))
    total_cost = []
    if cfg.future_term:
        for i in range(len(pred)):
            if pred[i] == 0:
                if label[i] != 0:
                    cost = cfg.cost.wrong_dm*n_steps
                else:
                    cost = cfg.cost.true_dm
            else:
                if label[i] == 0:
                    cost = cfg.cost.wrong_fl + 0.5*(cfg.cost.wrong_fl)*(n_steps - pred[i])
                elif pred[i] < label[i]:
                    cost = cfg.cost.early_fl + 0.5*(cfg.cost.early_fl)*(pred[i] - label[i])
                elif pred[i] == label[i]:
                    cost = cfg.cost.true_fl
                elif pred[i] > label[i]:
                    cost = cfg.cost.late_fl + 0.5*(cfg.cost.late_fl)*(label[i] - pred[i])
            total_cost.append(cost)
    else:
        for i in range(len(pred)):
            if pred[i] == 0:
                if label[i] != 0:
                    cost = cfg.cost.wrong_dm*n_steps
                else:
                    cost = cfg.cost.true_dm
            else:
                if label[i] == 0:
                    cost = cfg.cost.wrong_fl
                elif pred[i] < label[i]:
                    cost = cfg.cost.early_fl
                elif pred[i] == label[i]:
                    cost = cfg.cost.true_fl
                elif pred[i] > label[i]:
                    cost = cfg.cost.late_fl
            total_cost.append(cost)
    mean_cost = np.mean(total_cost)
    sum_cost = np.sum(total_cost)

    return mean_cost, sum_cost

def normalize_data(cfg,df):
    df = df