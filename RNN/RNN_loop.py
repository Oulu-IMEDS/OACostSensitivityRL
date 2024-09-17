import random

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


class RNNPipeline():
    def __init__(self, cfg, model, loss_func, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.best_loss = 10000
        self.best_mae = 10000
        self.best_cost = -10000
        self.best_roc_auc = -1

    def load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path), strict=True)

    def train_loop(self, train_loader, epoch_id):
        metrics_collector = {'loss': [], 'pred': [], 'label': []}
        encode_visit = ['0', '12', '24', '36', '48', '72', '96']
        # Tell the model that we are training it
        list_preds = []
        list_targets = []
        self.model.train(True)
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch_id}][Train]:")
        for batch_id, batch in enumerate(progress_bar):
            # Get sampled data and transfer them to the correct device
            loss = 0
            use_teacher_forcing = True if random.random() < self.cfg.teacher_forcing_ratio else False
            # use_teacher_forcing = 0.2
            input = batch[f'visit{encode_visit[0]}']
            input['KL_0'] = input['KL_0'].float()
            input['KL_1'] = input['KL_1'].float()
            h = torch.zeros((2 * self.cfg.RNN.n_layers, self.cfg.batch_size, self.cfg.RNN.hidden_size)).to(self.cfg.device)
            if use_teacher_forcing:
                for i in range(1, self.cfg.time_interval + 1):
                    # Forward through the model
                    _, h, _, preds = self.model(input, h)
                    target_KL_0 = batch[f'visit{encode_visit[i]}']['KL_0'].float()
                    target_KL_1 = batch[f'visit{encode_visit[i]}']['KL_1'].float()
                    pred_KL_0 = preds['KL_0'].squeeze()
                    pred_KL_1 = preds['KL_1'].squeeze()
                    loss_KL = self.loss_func['CE'](pred_KL_0, target_KL_0) + self.loss_func['CE'](pred_KL_1, target_KL_1)

                    target_WOMAC_0 = batch[f'visit{encode_visit[i]}']['WOMAC_0'].float()
                    target_WOMAC_1 = batch[f'visit{encode_visit[i]}']['WOMAC_1'].float()
                    pred_WOMAC_0 = preds['WOMAC_0'].squeeze()
                    pred_WOMAC_1 = preds['WOMAC_1'].squeeze()
                    loss_WOMAC = self.loss_func['MSE'](pred_WOMAC_0, target_WOMAC_0) + self.loss_func['MSE'](pred_WOMAC_1, target_WOMAC_1)

                    target_SF12 = batch[f'visit{encode_visit[i]}']['SF12'].float()
                    pred_SF12 = preds['SF12'].squeeze()
                    loss_SF12 = self.loss_func['MSE'](pred_SF12, target_SF12)

                    input = batch[f'visit{encode_visit[i]}']
                    input['KL_0'] = input['KL_0'].float()
                    input['KL_1'] = input['KL_1'].float()

                    list_preds.append(torch.argmax(pred_KL_0, dim=-1))
                    list_preds.append(torch.argmax(pred_KL_1, dim=-1))
                    list_preds.append(pred_WOMAC_0)
                    list_preds.append(pred_WOMAC_1)
                    list_preds.append(pred_SF12)

                    list_targets.append(torch.argmax(target_KL_0, dim=-1))
                    list_targets.append(torch.argmax(target_KL_1, dim=-1))
                    list_targets.append(target_WOMAC_0)
                    list_targets.append(target_WOMAC_1)
                    list_targets.append(target_SF12)

                    loss += loss_KL + loss_WOMAC + loss_SF12


            else:
                for i in range(1, self.cfg.time_interval + 1):
                    # Forward through the model
                    _, h, pred_input, preds = self.model(input, h)
                    target_KL_0 = batch[f'visit{encode_visit[i]}']['KL_0'].float()
                    target_KL_1 = batch[f'visit{encode_visit[i]}']['KL_1'].float()
                    pred_KL_0 = preds['KL_0'].squeeze()
                    pred_KL_1 = preds['KL_1'].squeeze()
                    loss_KL = self.loss_func['CE'](pred_KL_0, target_KL_0) + self.loss_func['CE'](pred_KL_1, target_KL_1)

                    target_WOMAC_0 = batch[f'visit{encode_visit[i]}']['WOMAC_0'].float()
                    target_WOMAC_1 = batch[f'visit{encode_visit[i]}']['WOMAC_1'].float()
                    pred_WOMAC_0 = preds['WOMAC_0'].squeeze()
                    pred_WOMAC_1 = preds['WOMAC_1'].squeeze()
                    loss_WOMAC = self.loss_func['MSE'](pred_WOMAC_0, target_WOMAC_0) + self.loss_func['MSE'](
                        pred_WOMAC_1, target_WOMAC_1)

                    target_SF12 = batch[f'visit{encode_visit[i]}']['SF12'].float()
                    pred_SF12 = preds['SF12'].squeeze()
                    loss_SF12 = self.loss_func['MSE'](pred_SF12, target_SF12)

                    input = pred_input

                    list_preds.append(torch.argmax(pred_KL_0, dim=-1))
                    list_preds.append(torch.argmax(pred_KL_1, dim=-1))
                    list_preds.append(pred_WOMAC_0)
                    list_preds.append(pred_WOMAC_1)
                    list_preds.append(pred_SF12)

                    list_targets.append(torch.argmax(target_KL_0, dim=-1))
                    list_targets.append(torch.argmax(target_KL_1, dim=-1))
                    list_targets.append(target_WOMAC_0)
                    list_targets.append(target_WOMAC_1)
                    list_targets.append(target_SF12)

                    loss += loss_KL + loss_WOMAC + loss_SF12


            # Set gradients to 0
            self.optimizer.zero_grad()

            # Calculate loss
            # loss = self.loss_func(preds, target)

            # Learn from the loss by applying backpropagation. This function will compute gradients
            loss.backward()
            # Update the model's weights based on the computed gradients and other parameters of the optimizer
            self.optimizer.step()

            # new_record = {'new_loss': loss.item(), 'new_preds': preds, 'new_labels': labels}
            # metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)
            metrics_collector['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
            metrics_collector['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()
            mae = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='mae')
            #
            # Update metrics to progress bar
            # metrics_display = {k: f'{metrics[k]:.03f}' for k in metrics}
            metrics_display = {'loss': loss.item() / self.cfg.time_interval, 'mae': mae}
            progress_bar.set_postfix(metrics_display)

    def eval_loop(self, eval_loader, epoch_id):
        metrics_collector = {'loss': [], 'pred': [], 'label': [], 'probs': [], 'balanced_accuracy': None}
        encode_visit = ['0', '12', '24', '36', '48', '72', '96']
        # Tell the model we are not training but evaluating it
        self.model.train(False)  # or model.eval()
        # Init dictionary to store metrics
        list_preds = []
        list_targets = []
        progress_bar = tqdm(eval_loader, total=len(eval_loader), desc=f"Epoch [{epoch_id}][Eval]:")
        with torch.no_grad():
            for batch_id, batch in enumerate(progress_bar):
                # Get sampled data and transfer them to the correct device
                display_metrics={}
                loss = 0
                input = batch[f'visit{encode_visit[0]}']
                input['KL_0'] = input['KL_0'].float()
                input['KL_1'] = input['KL_1'].float()
                h = torch.zeros((2 * self.cfg.RNN.n_layers, self.cfg.batch_size, self.cfg.RNN.hidden_size)).to(
                    self.cfg.device)
                for i in range(1, self.cfg.time_interval +1 ):
                    # Forward through the model
                    _, h, pred_input, preds = self.model(input, h)
                    target_KL_0 = batch[f'visit{encode_visit[i]}']['KL_0'].float()
                    target_KL_1 = batch[f'visit{encode_visit[i]}']['KL_1'].float()
                    pred_KL_0 = preds['KL_0'].squeeze()
                    pred_KL_1 = preds['KL_1'].squeeze()
                    loss_KL = self.loss_func['CE'](pred_KL_0, target_KL_0) + self.loss_func['CE'](pred_KL_1, target_KL_1)

                    target_WOMAC_0 = batch[f'visit{encode_visit[i]}']['WOMAC_0'].float()
                    target_WOMAC_1 = batch[f'visit{encode_visit[i]}']['WOMAC_1'].float()
                    pred_WOMAC_0 = preds['WOMAC_0'].squeeze()
                    pred_WOMAC_1 = preds['WOMAC_1'].squeeze()
                    loss_WOMAC = self.loss_func['MSE'](pred_WOMAC_0, target_WOMAC_0) + self.loss_func['MSE'](
                        pred_WOMAC_1, target_WOMAC_1)

                    target_SF12 = batch[f'visit{encode_visit[i]}']['SF12'].float()
                    pred_SF12 = preds['SF12'].squeeze()
                    loss_SF12 = self.loss_func['MSE'](pred_SF12, target_SF12)

                    input = pred_input

                    list_preds.append(torch.argmax(pred_KL_0, dim=-1))
                    list_preds.append(torch.argmax(pred_KL_1, dim=-1))
                    list_preds.append(pred_WOMAC_0)
                    list_preds.append(pred_WOMAC_1)
                    list_preds.append(pred_SF12)

                    list_targets.append(torch.argmax(target_KL_0, dim=-1))
                    list_targets.append(torch.argmax(target_KL_1, dim=-1))
                    list_targets.append(target_WOMAC_0)
                    list_targets.append(target_WOMAC_1)
                    list_targets.append(target_SF12)

                    # Calculate loss
                    loss += loss_KL + loss_WOMAC + loss_SF12

                display_metrics['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
                display_metrics['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()
                metrics_display = {'loss': loss.item() / self.cfg.time_interval,
                                   'mae': calculate_metrics_by_preds(self.cfg, display_metrics, metric_name='mae')}
                metric_loss = loss.item() / self.cfg.time_interval
                progress_bar.set_postfix(metrics_display)

            metrics_collector['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
            metrics_collector['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()
            # metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)
            metrics = OrderedDict()
            metrics['mae'] = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='mae')
            # metrics['roc-auc'] = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='roc-auc')

            # Store model based on balanced accuracy
            if metric_loss < self.best_loss:
                model_filename = f'pretrain_GRU_best_loss.pth'
                print(
                    f'\n Improved loss from {self.best_loss} to {metric_loss}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_loss = metric_loss

            if metrics['mae'] < self.best_mae:
                model_filename = f'pretrain_GRU_best_mae.pth'
                print(
                    f'\n Improved mae from {self.best_mae} to {metrics["mae"]}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_mae = metrics['mae']



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

class RNNPipelinePredictProgressionBaselineInput():
    def __init__(self, cfg, model, loss_func, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.best_loss = 10000
        self.best_mae = 10000
        self.best_cost = -10000
        self.best_roc_auc = -1
        self.best_ba = -1000

    def load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path), strict=True)

    def train_loop(self, train_loader, epoch_id):
        metrics_collector = {'loss': [], 'pred': [], 'label': []}
        encode_visit = ['0', '12', '24', '36', '48', '72', '96']
        # Tell the model that we are training it
        list_preds = []
        list_targets = []
        self.model.train(True)
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch_id}][Train]:")
        for batch_id, batch in enumerate(progress_bar):
            # Get sampled data and transfer them to the correct device
            loss = 0
            use_teacher_forcing = True if random.random() < self.cfg.teacher_forcing_ratio else False
            # use_teacher_forcing = 0.2
            input = batch[f'visit{encode_visit[0]}']
            input['KL_0'] = input['KL_0'].float()
            input['KL_1'] = input['KL_1'].float()
            if self.model.name == 'GRU':
                h = torch.zeros((2 * self.cfg.RNN.n_layers, self.cfg.batch_size, self.cfg.RNN.hidden_size)).to(self.cfg.device)
            elif self.model.name == 'LSTM':
                h_init = torch.zeros((2 * self.cfg.RNN.n_layers, self.cfg.batch_size, self.cfg.RNN.hidden_size)).to(
                    self.cfg.device)
                c_init = torch.zeros((2 * self.cfg.RNN.n_layers, self.cfg.batch_size, self.cfg.RNN.hidden_size)).to(
                    self.cfg.device)
                h = (h_init, c_init)

            for j in range(1, self.cfg.time_interval + 1):
                # Forward through the model
                i = 0
                pred, h = self.model(input, h)
                progress_label = batch[f'visit{encode_visit[j-1]}']['Progress']
                target = torch.ones((progress_label.shape[0],2))
                target[:,1] = target[:,1] * progress_label
                target[:, 0] = target[:, 0] - progress_label
                loss_time = self.loss_func['BCEWithLogits'](pred.squeeze(), target)

                loss += loss_time

                list_preds.append(torch.sigmoid(pred).detach().cpu().argmax(dim=-1))
                list_targets.append(progress_label.detach().cpu())

            # Set gradients to 0
            self.optimizer.zero_grad()

            # Calculate loss
            # loss = self.loss_func(preds, target)

            # Learn from the loss by applying backpropagation. This function will compute gradients
            loss.backward()
            # Update the model's weights based on the computed gradients and other parameters of the optimizer
            self.optimizer.step()

            # new_record = {'new_loss': loss.item(), 'new_preds': preds, 'new_labels': labels}
            # metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)
            metrics_collector['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
            metrics_collector['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()
            mae = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='mae')
            ba = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='balanced-accuracy')
            #
            # Update metrics to progress bar
            # metrics_display = {k: f'{metrics[k]:.03f}' for k in metrics}
            metrics_display = {'loss': loss.item() / self.cfg.time_interval, 'mae': mae, 'ba': ba}
            progress_bar.set_postfix(metrics_display)

    def eval_loop(self, eval_loader, epoch_id):
        metrics_collector = {'loss': [], 'pred': [], 'label': [], 'probs': [], 'balanced_accuracy': None}
        encode_visit = ['0', '12', '24', '36', '48', '72', '96']
        # Tell the model we are not training but evaluating it
        self.model.train(False)  # or model.eval()
        # Init dictionary to store metrics
        list_preds = []
        list_targets = []
        progress_bar = tqdm(eval_loader, total=len(eval_loader), desc=f"Epoch [{epoch_id}][Eval]:")
        with torch.no_grad():
            for batch_id, batch in enumerate(progress_bar):
                # Get sampled data and transfer them to the correct device
                display_metrics={}
                loss = 0
                input = batch[f'visit{encode_visit[0]}']
                input['KL_0'] = input['KL_0'].float()
                input['KL_1'] = input['KL_1'].float()
                # h = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(
                #     self.cfg.device)
                if self.model.name == 'GRU':
                    h = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(self.cfg.device)
                elif self.model.name == 'LSTM':
                    h_init = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(self.cfg.device)
                    c_init = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(self.cfg.device)
                    h = (h_init, c_init)
                for j in range(1, self.cfg.time_interval +1 ):
                    # Forward through the model
                    pred, h = self.model(input, h)
                    progress_label = batch[f'visit{encode_visit[j - 1]}']['Progress']
                    target = torch.ones((progress_label.shape[0], 2))
                    target[:, 1] = target[:, 1] * progress_label
                    target[:, 0] = target[:, 0] - progress_label
                    loss_time = self.loss_func['BCEWithLogits'](pred.squeeze(), target)

                    # Calculate loss
                    loss += loss_time

                    list_preds.append(torch.sigmoid(pred).detach().cpu().argmax(dim=-1))
                    list_targets.append(progress_label.detach().cpu())

                display_metrics['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
                display_metrics['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()
                metrics_display = {'loss': loss.item() / self.cfg.time_interval,
                                   'mae': calculate_metrics_by_preds(self.cfg, display_metrics, metric_name='mae'),
                                   'ba': calculate_metrics_by_preds(self.cfg, display_metrics, metric_name='balanced-accuracy'),
                                   }
                metric_loss = loss.item() / self.cfg.time_interval
                progress_bar.set_postfix(metrics_display)

            metrics_collector['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
            metrics_collector['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()
            # metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)
            metrics = OrderedDict()
            metrics['mae'] = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='mae')
            metrics['ba'] = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='balanced-accuracy')

            # Store model based on balanced accuracy
            if metric_loss < self.best_loss:
                model_filename = f'pretrain_{self.model.name}_best_loss_seed{self.cfg.seed}.pth'
                print(
                    f'\n Improved loss from {self.best_loss} to {metric_loss}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_loss = metric_loss

            if metrics['mae'] < self.best_mae:
                model_filename = f'pretrain_{self.model.name}_best_mae_seed{self.cfg.seed}.pth'
                print(
                    f'\n Improved mae from {self.best_mae} to {metrics["mae"]}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_mae = metrics['mae']

            if metrics['ba'] > self.best_ba:
                model_filename = f'pretrain_{self.model.name}_best_ba_seed{self.cfg.seed}.pth'
                print(
                    f'\n Improved ba from {self.best_ba} to {metrics["ba"]}. Saved to {model_filename}...')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_filename)
                self.best_ba = metrics['ba']



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
        encode_visit = ['0', '12', '24', '36', '48', '72', '96']
        # Tell the model we are not training but evaluating it
        self.model.eval()  # or model.eval()
        # Init dictionary to store metrics
        list_preds = []
        list_targets = []
        metrics = {}
        progress_bar = tqdm(test_loader, total=len(test_loader), desc=f"Epoch [0][Test]:")
        with torch.no_grad():
            for batch_id, batch in enumerate(progress_bar):
                # Get sampled data and transfer them to the correct device
                display_metrics={}
                loss = 0
                input = batch[f'visit{encode_visit[0]}']
                input['KL_0'] = input['KL_0'].float()
                input['KL_1'] = input['KL_1'].float()
                # h = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(
                #     self.cfg.device)
                if self.model.name == 'GRU':
                    h = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(self.cfg.device)
                elif self.model.name == 'LSTM':
                    h_init = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(self.cfg.device)
                    c_init = torch.zeros((2 * self.cfg.RNN.n_layers, input['AGE'].shape[0], self.cfg.RNN.hidden_size)).to(self.cfg.device)
                    h = (h_init, c_init)
                for j in range(1, self.cfg.time_interval +1 ):
                    # Forward through the model
                    pred, h = self.model(input, h)
                    progress_label = batch[f'visit{encode_visit[j - 1]}']['Progress']


                    list_preds.append(torch.sigmoid(pred).detach().cpu().argmax(dim=-1).squeeze().numpy())
                    list_targets.append(progress_label.detach().cpu())

                    metrics_collector['pred'] = torch.sigmoid(pred).detach().cpu().argmax(dim=-1)
                    metrics_collector['label'] = progress_label.detach().cpu()
                    _ba = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='balanced-accuracy')
                    _rc = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='recall')
                    _pr = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='precision')

                    print(f'Balance accuracy estimate progression after {j} year(s): {_ba}')
                    print(f'Precision score progression after {j} year(s): {_pr}')
                    print(f'Recall score progression after {j} year(s): {_rc}')

        return list_preds

                    # avg_ba += _ba

                # metric_loss = loss.item() / self.cfg.time_interval

            # metrics_collector['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
            # metrics_collector['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()    
            # # Store model based on balanced accuracy
            # metrics_collector['pred'] = torch.concat(list_preds, dim=0).detach().cpu().numpy()
            # metrics_collector['label'] = torch.concat(list_targets, dim=0).detach().cpu().numpy()
            # print(metrics_collector['pred'])
            # print(metrics_collector['label'])
            # _ba = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='balanced-accuracy')
            # mae = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='mae')
            # _roc_auc = calculate_metrics_by_preds(self.cfg, metrics_collector, metric_name='roc-auc')
            # cf_matrix = confusion_matrix(metrics_collector['label'], metrics_collector['pred'])

            # sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
            # plt.title(f'SL balance data')
            # plt.show()
            # print(cf_matrix)
            # print(f'Balanced Accuracy: {_ba}')
            # print(f'Mean Absolute error: {mae}')
            # print(f'ROC AUC score: {_roc_auc}')

            # mean_cost, sum_cost = convert_predictions_to_cost(self.cfg, label=metrics_collector['label'],
            #                                                   pred=metrics_collector['pred'])
            # print(f'Mean cost: {mean_cost}')
            # print(f'Total cost: {sum_cost}')

            # temp_preds = metrics_collector['pred']
            # temp_label = metrics_collector['label']
            # temp_preds = [1 if i > 0 else 0 for i in temp_preds]
            # temp_label = [1 if i > 0 else 0 for i in temp_label]

            # recall = recall_score(temp_label, temp_preds)
            # ba = balanced_accuracy_score(temp_label, temp_preds)
            # # mAP = average_precision_score(actions, targets, average='micro')
            # pr = precision_score(temp_label, temp_preds)
            # mae = mean_absolute_error(temp_label, temp_preds)
            # roc_auc = roc_auc_score(temp_label, metrics_collector['probs_binary'])
            # print(
            #     f'Balance accuracy binary: {ba} \n'
            #     f'Recall score binary: {recall} \n'
            #     f'Precision binary: {pr} \n'
            #     f'ROC AUC binary: {roc_auc} \n'

            # )
