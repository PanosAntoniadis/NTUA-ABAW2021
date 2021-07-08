import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

class resnet50_rnn(nn.Module):
    def __init__(self, num_classes, pretrained_affectnet, rnn_hidden_size, rnn_num_layers, dropout, rnn_type):
        super(resnet50_rnn, self).__init__()
        if pretrained_affectnet == "single":
            print("Loading pretrained single-task model on affectnet...")
            resnet50 = models.resnet50(pretrained=True)
            if num_classes == 7:
                checkpoint = torch.load('checkpoints_affectnet/resnet50_noAligned_wce_b64_augment')
                resnet50.fc = torch.nn.Linear(2048, 7)
                resnet50.load_state_dict(checkpoint['model_state_dict'])
            elif num_classes == 2:
                checkpoint = torch.load('checkpoints_affectnet/resnet50_noAligned_ccc_b64_augment_va')
                resnet50.fc = torch.nn.Linear(2048, 2)
                resnet50.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise NotImplementedError
                
            modules = list(resnet50.children())[:-1]
        else:
            resnet50 = models.resnet50(pretrained=True)
            modules = list(resnet50.children())[:-1]      # delete the last fc layer.

        self.features = nn.Sequential(*modules)
        num_features = 2048
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)
        self.rnn_type = rnn_type
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        y = self.features(x)
        y = y.view(b, t, -1)
        out, _ = self.rnn(y)

        out = out.reshape(b*t, -1)
        output = self.fc(out)
        return output

