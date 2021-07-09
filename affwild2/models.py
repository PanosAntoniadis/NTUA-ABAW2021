import json
import torch
import torchvision
import torchvision.models as models

from torch import nn

class resnet50_rnn(nn.Module):
    
    def __init__(self, num_classes, pretrained_affectnet, rnn_hidden_size, rnn_num_layers, 
                 dropout, rnn_type, body_arch, context_arch, bidirectional, body=False, context=False, audio=False):
        
        super(resnet50_rnn, self).__init__()

        self.bidirectional = bidirectional
        self.body = body
        self.audio = audio
        self.context = context

        if self.body:
            # Build body backbone
            _model = getattr(torchvision.models, body_arch)(pretrained = True)
            self.num_body_features = _model.fc.in_features
            _modules = list(_model.children())[:-1] # delete the last fc layer.
            self.body_model = nn.Sequential(*_modules)

        if self.context:
            places_model_file = '/gpu-data2/jpik/%s_places365.pth.tar' % context_arch
            places_model = torchvision.models.__dict__[context_arch](num_classes=365)
            _checkpoint = torch.load(places_model_file, map_location=lambda storage, loc: storage)
            #print(_checkpoint)
            state_dict = {str.replace(k,'module.',''): v for k, v in _checkpoint['state_dict'].items()}
            #print(state_dict)
            places_model.load_state_dict(state_dict)
            self.num_context_features = places_model.fc.in_features
            __modules_ = list(places_model.children())[:-1]  # delete the last fc layer.
            self.context_model = nn.Sequential(*__modules_)

        # Build face backbone         
        resnet50 = models.resnet18(pretrained=True)
        if pretrained_affectnet:
            print('Getting affectnet pretrained')
            checkpoint = torch.load('/home/filby/abaw_2021/checkpoints_affectnet/resnet50_noAligned_wce_b64_augment')
            resnet50.fc = torch.nn.Linear(2048, 7)
            resnet50.load_state_dict(checkpoint['model_state_dict'])

        modules = list(resnet50.children())[:-1]  # delete the last fc layer.
        self.face_model = nn.Sequential(*modules)


        num_features = resnet50.fc.in_features

        if self.context:
            num_features += self.num_context_features

        if self.body:
            num_features += self.num_body_features

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout,
                               bidirectional=self.bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        
        if self.bidirectional:
            self.fc = nn.Linear(2*rnn_hidden_size, num_classes)
        else:
            self.fc = nn.Linear(rnn_hidden_size, num_classes)       
        self.rnn_type = rnn_type
        
    def forward(self, x_face, x_body=None, x_context=None, audio=None):
        b, t, c, h, w = x_face.shape
        
        x_face = x_face.view(b*t, c, h, w)
        y_face = self.face_model(x_face)
        y_face = y_face.view(b, t, -1)

        if self.body:
            x_body = x_body.view(b*t, c, h, w)
            y_body = self.body_model(x_body)
            y_body = y_body.view(b, t, -1)
        if self.context:
            x_context = x_context.view(b*t, c, h, w)
            y_context = self.context_model(x_context)
            y_context = y_context.view(b, t, -1)
        
        if self.body and self.context:
            y = torch.cat((y_face, y_body, y_context), dim = 2)
        else:
            y = y_face

        self.rnn.flatten_parameters()

        if self.rnn_type == 'transformer':
            y = y.transpose(0,1)
            out = self.rnn(y)
            out = out.transpose(0,1)
        else:
            out, _ = self.rnn(y)

        out = out.reshape(b*t, -1)
        output = self.fc(out)
        return output

    def get_optim_policies(self):
        params = [{'params': self.parameters()}]
        return params
