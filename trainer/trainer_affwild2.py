import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, make_barplot, features_blobs, setup_cam, returnCAM
import matplotlib as mpl
import random

mpl.use('Agg')
import matplotlib.pyplot as plt
import model.metric
import model.loss
import torch.nn.functional as F

def mse_center_loss_expr(output, target, labels):
    t = labels.clone().detach()

    target = target[0, :7]

    positive_centers = target[labels]

    # print(positive_centers.shape, target.shape, output.shape)

    loss = F.mse_loss(output, positive_centers)

    return loss


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metrics, optimizer, config, data_loader,
                 track, valid_data_loader=None, lr_scheduler=None, len_epoch=None, optical_flow=False):
        super().__init__(model, criterion, metrics, optimizer, config)
        self.data_loader = data_loader
        self.track = track
        self.optical_flow = optical_flow
        self.body = False
        self.audio = True
        self.context = False
        self.embed = False

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(self.len_epoch/10)

        self.criterion = criterion
        
        if self.track == 1:
            self.train_metrics = MetricTracker('ccc_v', 'ccc_a', 'total', 'loss', writer=self.writer)
            self.valid_metrics = MetricTracker('ccc_v', 'ccc_a', 'total', 'loss', writer=self.writer)
            self.num_classes = 2
        elif self.track == 2:
            self.train_metrics = MetricTracker('accuracy', 'f1_score', 'total', 'loss', writer=self.writer)
            self.valid_metrics = MetricTracker('accuracy', 'f1_score', 'total', 'loss', writer=self.writer)
            self.num_classes = 7
        elif self.track == 3:
            self.train_metrics = MetricTracker('accuracy_binary', 'f1_score_binary', 'total', 'loss', writer=self.writer)
            self.valid_metrics = MetricTracker('accuracy_binary', 'f1_score_binary', 'total', 'loss', writer=self.writer)
            self.num_classes = 12
        else:
            raise NotImplementedError
            

    def _train_epoch(self, epoch, phase="train"):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.logger.info('Current LR is')
        for param_group in self.optimizer.param_groups:
            self.logger.info('{}'.format(param_group['lr']))

        if phase == "train":
            self.model.train()
            self.train_metrics.reset()
            torch.set_grad_enabled(True)
            metrics = self.train_metrics
        elif phase == "val" :
            self.model.eval()
            self.valid_metrics.reset()
            torch.set_grad_enabled(False)
            metrics = self.valid_metrics

        outputs = []
        targets = []

        data_loader = self.data_loader if phase == "train" else self.valid_data_loader

        running_loss = 0.0
        count = 0


        for batch_idx, data in enumerate(data_loader):
            # print(audio)
            if self.track == 1:
                target = data['cont']
            else:
                target = data['expressions']

            valid = data['valid']

            if self.embed:
                embeddings = data['embeddings'].to(self.device)

            if self.optical_flow:
                faces = data['flow']
                faces = faces.to(self.device)
                b, t, c, h, w = faces.shape
            else:
                if not self.audio:
                    faces = data['faces']
                    faces = faces.to(self.device)
                    b, t, c, h, w = faces.shape

            if self.audio:
                audio = data['audio'].to(self.device)
                # print(audio.shape)
                b, t, c, h, w = audio.shape

            target = target.to(self.device)

            if self.body and self.context:
                contexts = data['context']
                bodies = data['body']
                bodies, contexts = bodies.to(self.device), contexts.to(self.device)

            if phase == "train":
                self.optimizer.zero_grad()

            # out = self.model(faces, bodies, contexts)
            out= self.model(audio)
            # out, out_embed = self.model(audio)
            out = out.view(b*t, self.num_classes)


            target = target.view(b*t, -1).squeeze()
            valid = valid.view(b*t)
            if phase == "val":
                # On validation phase some frames are invalid 
                # int order to have equal duration per sample.
                out = out[valid]
                target = target[valid]
                # out_embed = out_embed[valid]
            loss = self.criterion(out, target)

            if self.embed:
                loss_embed = mse_center_loss_expr(out_embed, embeddings, target)
                loss += loss_embed

            if phase == "train":
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()*out.size(0)
            count += out.size(0)

            if self.track == 3:
                out = torch.sigmoid(out)
            outputs.append(out.cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy())

            if batch_idx % self.log_step == 0:
                self.logger.debug('{} Epoch: {} {} Loss: {:.6f}'.format(
                    phase,
                    epoch,
                    self._progress(batch_idx, data_loader, phase),
                    running_loss/count))

            if batch_idx == self.len_epoch and phase == "train":
                break

        if phase == "train":
            self.writer.set_step(epoch)
        else:
            self.writer.set_step(epoch, phase)

        # assert(count==len(data_loader.dataset))
        epoch_loss = running_loss/count
        # print(count)
        metrics.update('loss', epoch_loss)


        output = np.concatenate(outputs, axis=0)
        target = np.concatenate(targets, axis=0)
        
        if self.track == 1:
            ccc_v, ccc_a = model.metric.ccc(output, target)
            metrics.update("ccc_v", ccc_v)
            metrics.update("ccc_a", ccc_a)
            total = (ccc_v + ccc_a) /2
            metrics.update("total", total)
        elif self.track == 2:
            accuracy = model.metric.accuracy(output, target)
            f1_score = model.metric.f1_score(output, target)
            metrics.update("accuracy", accuracy)
            metrics.update("f1_score", f1_score)
            total = 0.67*f1_score + 0.33*accuracy
            metrics.update("total", total)

            # f1_score_all = model.metric.f1_score(output, target, average=None)
            # self.writer.add_figure('%s f1 score per class' % phase,
            #                        make_barplot(f1_score_all, ['Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise'], 'f1 score'))

        elif self.track == 3:
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            output = output.flatten()
            target = target.flatten()
            accuracy_binary = model.metric.accuracy_binary(output, target)
            f1_score_binary = model.metric.f1_score_binary(output, target)
            metrics.update("accuracy_binary", accuracy_binary)
            metrics.update("f1_score_binary", f1_score_binary)
            total = 0.5*f1_score_binary + 0.5*accuracy_binary
            metrics.update("total", total)
        else:
            raise NotImplementedError

        log = metrics.result()

        if phase == "train":
            if self.do_validation:
                val_log = self._train_epoch(epoch, phase="val")
                log.update(**{'val_' + k: v for k, v in val_log.items()})

            return log

        elif phase == "val":
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(total)
                else:
                    self.lr_scheduler.step()

            self.writer.save_results(output, "output")
            self.writer.save_results(target, "target")

            return metrics.result()


    def _progress(self, batch_idx, data_loader, phase):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            if phase == 'train':
                total = self.len_epoch
            else:
                total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)
