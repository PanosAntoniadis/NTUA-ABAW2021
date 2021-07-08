import json
import torch
import torchvision
from torch import nn
import torchvision.models as models

class resnet50_body_rnn(nn.Module):
    def __init__(self, num_classes, pretrained_affectnet, rnn_hidden_size, rnn_num_layers, 
                 dropout, rnn_type, body_arch, bidirectional):
        
        super(resnet50_body_rnn, self).__init__()
        self.bidirectional = bidirectional
        # Build body backbone
        _model = getattr(torchvision.models, body_arch)(pretrained = True)
        self.num_body_features = _model.fc.in_features
        _modules = list(_model.children())[:-1] # delete the last fc layer.
        self.body_model = nn.Sequential(*_modules)                      
        
        # Build face backbone         
        resnet50 = models.resnet50(pretrained=True)
        if pretrained_affectnet:
            checkpoint = torch.load('/glaros_home/filby/abaw_2021/checkpoints_affectnet/resnet50_noAligned_wce_b64_augment')
            resnet50.fc = torch.nn.Linear(2048, 7)
            resnet50.load_state_dict(checkpoint['model_state_dict'])
        num_features = resnet50.fc.in_features + self.num_body_features
        modules = list(resnet50.children())[:-1]  # delete the last fc layer.
        self.face_model = nn.Sequential(*modules)
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)

        if self.bidirectional:
            self.fc = nn.Linear(2*rnn_hidden_size, num_classes)
        else:
            self.fc = nn.Linear(rnn_hidden_size, num_classes)       
        self.rnn_type = rnn_type
        
    def forward(self, x_face, x_body):
        b, t, c, h, w = x_face.shape
        
        x_face = x_face.view(b*t, c, h, w)
        x_body = x_body.view(b*t, c, h, w)
        
        y_face = self.face_model(x_face)
        y_body = self.body_model(x_body)
        
        y_face = y_face.view(b, t, -1)
        y_body = y_body.view(b, t, -1)
        
        if self.body:
            y = torch.cat((y_face, y_body), dim = 2)
        else:
            y = y_face    
        
        out, _ = self.rnn(y)

        out = out.reshape(b*t, -1)
        output = self.fc(out)
        return output

    def get_optim_policies(self):
        params = [{'params': self.parameters()}]
        return params


class resnet50_bc_rnn(nn.Module):
    def __init__(self, num_classes, pretrained_affectnet, rnn_hidden_size, rnn_num_layers, 
                 dropout, rnn_type, body_arch, context_arch, bidirectional):  
        super(resnet50_bc_rnn, self).__init__()

        self.bidirectional = bidirectional
        
        # Build body backbone
        _model = getattr(torchvision.models, body_arch)(pretrained = True)
        self.num_body_features = _model.fc.in_features
        _modules = list(_model.children())[:-1] # delete the last fc layer.
        self.body_model = nn.Sequential(*_modules)                      
        
        places_model_file = '/gpu-data2/jpik/%s_places365.pth.tar' % context_arch
        places_model = torchvision.models.__dict__[context_arch](num_classes=365)
        _checkpoint = torch.load(places_model_file, map_location=lambda storage, loc: storage)

        state_dict = {str.replace(k,'module.',''): v for k, v in _checkpoint['state_dict'].items()}

        places_model.load_state_dict(state_dict)
        self.num_context_features = places_model.fc.in_features
        __modules_ = list(places_model.children())[:-1]  # delete the last fc layer.
        self.context_model = nn.Sequential(*__modules_) 
        
        # Build face backbone         
        resnet50 = models.resnet50(pretrained=True)
        if pretrained_affectnet:
            checkpoint = torch.load('/glaros_home/filby/abaw_2021/checkpoints_affectnet/resnet50_noAligned_wce_b64_augment')
            resnet50.fc = torch.nn.Linear(2048, 7)
            resnet50.load_state_dict(checkpoint['model_state_dict'])
        num_features = resnet50.fc.in_features + self.num_body_features + self.num_context_features
        modules = list(resnet50.children())[:-1]  # delete the last fc layer.
        self.face_model = nn.Sequential(*modules)
        
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
        
    def forward(self, x_face, x_body, x_context, audio=None):
        b, t, c, h, w = x_face.shape
        
        x_face = x_face.view(b*t, c, h, w)
        x_body = x_body.view(b*t, c, h, w)
        x_context = x_context.view(b*t, c, h, w)

        y_face = self.face_model(x_face)
        y_body = self.body_model(x_body)
        y_context = self.context_model(x_context)        
        
        y_face = y_face.view(b, t, -1)
        y_body = y_body.view(b, t, -1)
        y_context = y_context.view(b, t, -1)
        
        # if self.body and self.context:
        y = torch.cat((y_face, y_body, y_context), dim = 2)
        # else:
        #     y = y_face
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
    

class resnet50_context_rnn(nn.Module): 
    def __init__(self, num_classes, pretrained_affectnet, rnn_hidden_size, rnn_num_layers, 
                 dropout, rnn_type, context_arch, bidirectional):
        
        super(resnet50_context_rnn, self).__init__()
        self.bidirectional = bidirectional

        # Build context backbone
        places_model_file = '/gpu-data2/jpik/%s_places365.pth.tar' % context_arch
        places_model = torchvision.models.__dict__[context_arch](num_classes=365)
        _checkpoint = torch.load(places_model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k, v in _checkpoint['state_dict'].items()}
        places_model.load_state_dict(state_dict)
        self.num_context_features = places_model.fc.in_features
        _modules = list(places_model.children())[:-1]  # delete the last fc layer.
        self.context_model = nn.Sequential(*_modules)                     
        
        # Build face backbone         
        resnet50 = models.resnet50(pretrained=True)
        if pretrained_affectnet:
            checkpoint = torch.load('/glaros_home/filby/abaw_2021/checkpoints_affectnet/resnet50_noAligned_wce_b64_augment')
            resnet50.fc = torch.nn.Linear(2048, 7)
            resnet50.load_state_dict(checkpoint['model_state_dict'])
        num_features = resnet50.fc.in_features + self.num_context_features
        modules = list(resnet50.children())[:-1]  # delete the last fc layer.
        self.face_model = nn.Sequential(*modules)
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, dropout=dropout)
        if self.bidirectional:
            self.fc = nn.Linear(2*rnn_hidden_size, num_classes)
        else:
            self.fc = nn.Linear(rnn_hidden_size, num_classes)       
        self.rnn_type = rnn_type
        
    def forward(self, x_face, x_context):
        b, t, c, h, w = x_face.shape
        
        x_face = x_face.view(b*t, c, h, w)
        x_context = x_context.view(b*t, c, h, w)
        
        y_face = self.face_model(x_face)
        y_context = self.context_model(x_context)
        
        y_face = y_face.view(b, t, -1)
        y_context = y_context.view(b, t, -1)
        
        if self.context:
            y = torch.cat((y_face, y_context), dim = 2)
        else:
            y = y_face
        
        out, _ = self.rnn(y)

        out = out.reshape(b*t, -1)
        output = self.fc(out)
        return output

    def get_optim_policies(self):
        params = [{'params': self.parameters()}]
        return params    