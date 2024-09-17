import numpy as np

from Supervised_Learning.utils import collect_and_update_metrics, to_cpu, calculate_metrics_by_preds, \
    convert_predictions_to_cost
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, mean_absolute_error, roc_auc_score, \
    confusion_matrix
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


class SupervisedLearningPipeline():
    def __init__(self, cfg, model, loss_func, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.best_bacc = -1
        self.best_cost = -10000
        self.best_roc_auc = -1

    def load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path), strict=True)

    def train_loop(self, train_loader, epoch_id):
        metrics_collector = {'loss': [], 'pred': [], 'label': []}
        # Tell the model that we are training it
        self.model.train(True)
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch_id}][Train]:")
        for batch_id, batch in enumerate(progress_bar):
            # Get sampled data and transfer them to the correct device
            if self.cfg.beam_data:
                inputs = batch['data']
            else:
                inputs = batch['data'].to(self.device)
            labels = batch['target'].to(self.device)

            # Forward through the model
            preds = self.model(inputs)

            # Set gradients to 0
            self.optimizer.zero_grad()

            # Calculate loss
            loss = self.loss_func(preds, labels)

            # Learn from the loss by applying backpropagation. This function will compute gradients
            loss.backward()
            # Update the model's weights based on the computed gradients and other parameters of the optimizer
            self.optimizer.step()

            new_record = {'new_loss': loss.item(), 'new_preds': preds, 'new_labels': labels}
            # metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)
            #
            # Update metrics to progress bar
            # metrics_display = {k: f'{metrics[k]:.03f}' for k in metrics}
            metrics_display = {'loss': loss.item()}
            progress_bar.set_postfix(metrics_display)

    def eval_loop(self, eval_loader, epoch_id):
        metrics_collector = {'loss': [], 'pred': [], 'label': [], 'probs': [], 'balanced_accuracy': None}
        # Tell the model we are not training but evaluating it
        self.model.train(False)  # or model.eval()
        # Init dictionary to store metrics
        metrics = OrderedDict()
        progress_bar = tqdm(eval_loader, total=len(eval_loader), desc=f"Epoch [{epoch_id}][Eval]:")
        with torch.no_grad():
            for batch_id, batch in enumerate(progress_bar):
                # Get sampled data and transfer them to the correct device
                if self.cfg.beam_data:
                    inputs = batch['data']
                else:
                    inputs = batch['data'].to(self.device)
                labels = batch['target'].to(self.device)

                # Forward through the model
                preds = self.model(inputs)

                if not self.cfg.progression_within:
                    if self.cfg.time_interval == 1:
                        metrics_collector['probs'] += nn.Softmax(dim=1)(preds)[:, 1].tolist()
                    elif self.cfg.time_interval > 1:
                        metrics_collector['probs'] += nn.Softmax(dim=1)(preds).tolist()
                    np_preds_classes = to_cpu(torch.argmax(preds, dim=1))
                else:
                    metrics_collector['probs'] += to_cpu(preds).tolist()
                    np_preds_classes = (preds > 0.5)*1

                # Calculate loss
                loss = self.loss_func(preds, labels)

                # Calculate loss

                np_labels = to_cpu(labels)
                metrics_collector['pred'] += np_preds_classes.tolist()
                metrics_collector['label'] += np_labels.tolist()

                metrics_display = {'loss': loss.item(),
                                   'balanced_accuracy': calculate_metrics_by_preds(self.cfg, metrics_collector,
                                                                                   metric_name='balanced-accuracy'),
                                   'roc_auc': calculate_metrics_by_preds(self.cfg, metrics_collector,
                                                                         metric_name='roc-auc')}
                progress_bar.set_postfix(metrics_display)

            # metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)
            metrics = OrderedDict()
            metrics['balanced_accuracy'] = calculate_metrics_by_preds(self.cfg, metrics_collector,
                                                                      metric_name='balanced-accuracy')
            metrics['cost'] = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='cost')
            metrics['roc-auc'] = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='roc-auc')

            # Store model based on balanced accuracy
            if metrics['balanced_accuracy'] > self.best_bacc:
                model_filename = f'oai_best_bacc_{self.cfg.set_features}{self.cfg.fix_features}.pth'
                print(
                    f'\n Improved balanced accuracy from {self.best_bacc} to {metrics["balanced_accuracy"]}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_bacc = metrics['balanced_accuracy']

            if metrics['cost'] > self.best_cost:
                model_filename = f'oai_best_cost_{self.cfg.set_features}{self.cfg.fix_features}.pth'
                print(
                    f'\n Improved cost from {self.best_cost} to {metrics["cost"]}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_cost = metrics['cost']

            if metrics['roc-auc'] > self.best_roc_auc:
                model_filename = f'oai_best_roc_auc_{self.cfg.set_features}{self.cfg.fix_features}.pth'
                print(
                    f'\n Improved roc-auc from {self.best_roc_auc} to {metrics["roc-auc"]}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_roc_auc = metrics['roc-auc']

            # Store model based on mean absolute error
            # if metrics['mean_abs_err'] < best_mae:
            #     model_filename = "oai_best_mae.pth"
            #     print(
            #         f'Improved mean absolute error from {best_mae} to {metrics["mean_abs_err"]}. Saved to {model_filename}...')
            #     torch.save(self.model.state_dict(), model_filename)
            #     best_mae = metrics['mean_abs_err']

    def test_loop(self, test_loader):
        metrics_collector = {'loss': [], 'pred': [], 'label': [], 'probs': [], 'balanced_accuracy': None,
                             'probs_binary': []}
        # Tell the model we are not training but evaluating it
        self.model.eval()  # or model.eval()
        # Init dictionary to store metrics
        metrics = {}
        progress_bar = tqdm(test_loader, total=len(test_loader), desc=f"Epoch [0][Test]:")
        with torch.no_grad():
            for batch_id, batch in enumerate(progress_bar):
                # Get sampled data and transfer them to the correct device
                if self.cfg.beam_data:
                    inputs = batch['data']
                else:
                    inputs = batch['data'].to(self.device)
                labels = batch['target'].to(self.device)

                # Forward through the model
                preds = self.model(inputs)
                if self.cfg.progression_within == False:
                    if self.cfg.time_interval == 1:
                        metrics_collector['probs'] += nn.Softmax(dim=1)(preds)[:, 1].tolist()
                    elif self.cfg.time_interval > 1:
                        metrics_collector['probs'] += nn.Softmax(dim=1)(preds).tolist()
                    np_preds_classes = to_cpu(torch.argmax(preds, dim=1))
                else:
                    metrics_collector['probs'] += to_cpu(preds).tolist()
                    np_preds_classes = (preds > 0.5)*1


                # Calculate loss
                np_labels = to_cpu(labels)
                metrics_collector['pred'] += np_preds_classes.tolist()
                metrics_collector['label'] += np_labels.tolist()

            # Store model based on balanced accuracy
            print(metrics_collector['pred'])
            print(metrics_collector['label'])
            _ba = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='balanced-accuracy')
            mae = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='mae')
            _roc_auc = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='roc-auc')
            cf_matrix = confusion_matrix(metrics_collector['label'], metrics_collector['pred'])

            sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
            plt.title(f'SL balance data')
            plt.show()
            print(cf_matrix)
            print(f'Balanced Accuracy: {_ba}')
            print(f'Mean Absolute error: {mae}')
            print(f'ROC AUC score: {_roc_auc}')

            mean_cost, sum_cost = convert_predictions_to_cost(self.cfg, label=metrics_collector['label'],
                                                              pred=metrics_collector['pred'])
            print(f'Mean cost: {mean_cost}')
            print(f'Total cost: {sum_cost}')

            temp_preds = metrics_collector['pred']
            temp_label = metrics_collector['label']
            temp_preds = [1 if i > 0 else 0 for i in temp_preds]
            temp_label = [1 if i > 0 else 0 for i in temp_label]

            recall = recall_score(temp_label, temp_preds)
            ba = balanced_accuracy_score(temp_label, temp_preds)
            # mAP = average_precision_score(actions, targets, average='micro')
            pr = precision_score(temp_label, temp_preds)
            mae = mean_absolute_error(temp_label, temp_preds)
            roc_auc = roc_auc_score(temp_label, metrics_collector['probs_binary'])
            print(
                f'Balance accuracy binary: {ba} \n'
                f'Recall score binary: {recall} \n'
                f'Precision binary: {pr} \n'
                f'ROC AUC binary: {roc_auc} \n'

            )
