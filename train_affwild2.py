import argparse
import collections
import torch
from torchvision import transforms, models
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from logger import setup_logging
from model import loss
# from affwild2.dataset import Video_dataset_cat, Video_dataset_cont, Video_dataset_au, Video_dataset_mtl
# from affwild2.dataset_eval import Video_dataset_cat_eval, Video_dataset_cont_eval, Video_dataset_au_eval

from affwild2.dataset_bc import Video_dataset_cat, Video_dataset_cont

from affwild2.dataset_eval_bc import Video_dataset_cat_eval, Video_dataset_cont_eval, Video_dataset_au_eval


from trainer.trainer_affwild2_audio import Trainer
from trainer.trainer_affwild2_mtl import Trainer_mtl
from trainer.trainer_affwild2_mtl_separate import Trainer_mtl_separate
from affwild2.models_combination import resnet50_rnn

from affwild2.gcn_models import resnet50_rnn_gcn

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args, config):

    batch_size = config['data_loader']['batch_size']
    num_workers = config['data_loader']['num_workers']
    data_pkl = config['data_loader']['data_pkl']
    duration = config['data_loader']['duration']
    track = config['track']  # 1:VA / 2:EXPR / 3:AU 

    train_transform = torchvision.transforms.Compose([
        transforms.Resize(size=(112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])])
    val_transform = torchvision.transforms.Compose([
        transforms.Resize(size=(112, 112)),                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])])

    if track == 1:
        # valence-arousal estimation
        num_classes = 2
        val_dataset = Video_dataset_cont(data_pkl, train=False, transform=val_transform, duration=duration, audio=config['modalities']['audio'], context=config['modalities']['context'], body=config['modalities']['body'], optical_flow=config['modalities']['optical_flow'])
        metrics = [getattr(module_metric, met) for met in config['metrics_continuous']]
        total_frames = 1593961
        train_dataset = Video_dataset_cont(data_pkl, train=True, transform=train_transform, duration=duration,
                                            audio=config['modalities']['audio'],
                                            context=config['modalities']['context'], body=config['modalities']['body'],
                                            optical_flow=config['modalities']['optical_flow'])
        criterion = getattr(module_loss, config['loss_continuous'])
    elif track == 2: 
        # seven basic expression classification
        num_classes = 7
        val_dataset = Video_dataset_cat(data_pkl, train=False, transform=val_transform, duration=duration, audio=config['modalities']['audio'], context=config['modalities']['context'], body=config['modalities']['body'], optical_flow=config['modalities']['optical_flow'])
        metrics = [getattr(module_metric, met) for met in config['metrics_categorical']]
        total_frames = 557154
        train_dataset = Video_dataset_cat(data_pkl, train=True, transform=train_transform, duration=duration, audio=config['modalities']['audio'], context=config['modalities']['context'], body=config['modalities']['body'], optical_flow=config['modalities']['optical_flow'])
        criterion = getattr(module_loss, config['loss_categorical'])        
    else:
        raise  NotImplementedError


    print("Total train data:{}".format(len(train_dataset)))
    print("Iterations per epoch:{}".format(len_epoch))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    print("Total val data:{}".format(len(val_dataset)))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)


    if config['arch']['type'] == 'resnet50_rnn':
        model = resnet50_rnn(num_classes=num_classes,
                                   pretrained_affectnet=config['arch']['args']['pretrained_affectnet'],
                                   rnn_hidden_size=config['rnn']['hidden_size'],
                                   rnn_num_layers=config['rnn']['num_layers'],
                                   dropout=config['arch']['args']['dropout'], rnn_type=config['rnn']['type'],
                                audio=config['modalities']['audio'], context=config['modalities']['context'],
                                body=config['modalities']['body'], optical_flow=config['modalities']['optical_flow'],
                                   body_arch='resnet50', context_arch='resnet50', bidirectional=True)
    else:
        raise NotImplementedError

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    logger = config.get_logger('train')
    logger.info(model)



    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                    config=config,
                    data_loader=train_loader,
                    track=track,
                    valid_data_loader=val_loader,
                    lr_scheduler=lr_scheduler, len_epoch=len_epoch, optical_flow=config['modalities']['optical_flow'])

    trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning model on Aff-Wild 2 for ABAW 2021')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                        metavar='N', help='evaluation frequency (default: 5)')

    # ========================= Runtime Configs ==========================

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp_name'], type=str, target='name'),

        # ========================= Task Configs ==========================

        CustomArgs(['--track'], type=int, target='track'),

        # ========================= Model Configs ==========================

        CustomArgs(['--arch'], type=str, target='arch;type'),
        CustomArgs(['--dropout'], type=float, target='arch;args;dropout'),
        CustomArgs(['--pretrained_affectnet'], type=bool, target='arch;args;pretrained_affectnet'),

        CustomArgs(['--cell_type'], type=str, target='rnn;type'),
        CustomArgs(['--hidden_size'], type=int, target='rnn;hidden_size'),
        CustomArgs(['--num_layers'], type=int, target='rnn;num_layers'),

        CustomArgs(['--context'], type=bool, target='modalities;context'),
        CustomArgs(['--body'], type=bool, target='modalities;body'),
        CustomArgs(['--face'], type=bool, target='modalities;face'),
        CustomArgs(['--audio'], type=bool, target='modalities;audio'),

        # ========================= Optimizer Configs ==========================

        CustomArgs(['--optimizer'], type=str, target="optimizer;type"),

        CustomArgs(['--lr', '--learning_rate'], type=float, target="optimizer;args;lr"),
        CustomArgs(['--momentum'], type=float, target="optimizer;args;momentum"),
        CustomArgs(['--weight_decay', '--wd'], type=float, target="optimizer;args;weight_decay"),


        CustomArgs(['--duration'], type=int, target="data_loader;duration"),
        CustomArgs(['--batch_size'], type=int, target="data_loader;batch_size")
    ]
    config = ConfigParser.from_args(parser, options)

    args = parser.parse_args()

    main(args, config)


