import torch
import torch.nn as nn
import torchvision
from model import backbone, mmd

class Base_Net(nn.Module):
    def __init__(self, num_class, base_net='resnet18', width=1024):
        super(Base_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        classifier_layer_list = [
            nn.Linear(self.base_network.output_num(), width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_class)
            ]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, data):
        feature = self.base_network(data)
        clf = self.classifier_layer(feature)
        return clf

    def predict(self, data):
        return self.forward(data)

class Transfer_Net(nn.Module):
    def __init__(self, num_class, base_net='resnet18', transfer_loss='mmd', use_domain_loss=True, width=1024, bottleneck_width=256):
        super(Transfer_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]()   # the last layer is the average pooling
        self.use_domain_loss = use_domain_loss
        self.transfer_loss = transfer_loss

        self.fc_layers = nn.Sequential(
            *[nn.Linear(self.base_network.output_num(), width), nn.BatchNorm1d(width), nn.ReLU()])
        self.bottleneck_layer = nn.Sequential(
            *[nn.Linear(width, bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)])
        self.classifier_layer = nn.Sequential(
            *[nn.Linear(bottleneck_width, num_class)])

        # initilization
        self.fc_layers[0].weight.data.normal_(0, 0.005)
        self.fc_layers[0].bias.data.fill_(0.1)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        self.classifier_layer[0].weight.data.normal_(0, 0.01)
        self.classifier_layer[0].bias.data.fill_(0.0)

    def forward(self, source, target):
        source = self.bottleneck_layer(self.fc_layers(self.base_network(source)))
        source_clf = self.classifier_layer(source)
        if self.use_domain_loss:
            target = self.bottleneck_layer(self.fc_layers(self.base_network(target)))
            transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
            return source_clf, transfer_loss
        else:
            return source_clf

    def predict(self, x):
        x = self.bottleneck_layer(self.fc_layers(self.base_network(x)))
        clf = self.classifier_layer(x)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        else:
            loss = 0
        return loss


class GRL_Net(nn.Module):
    def __init__(self, num_class, base_net='resnet18',  width=1024):
        super(GRL_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]()   # the last layer is the average pooling
        self.fc_layers = nn.Sequential(
            *[nn.Linear(self.base_network.output_num(), width), nn.BatchNorm1d(width), nn.LeakyReLU()]
            )
        self.classifier_layer = nn.Sequential(
            *[nn.Linear(width, num_class)])
        self.domain_discriminator = nn.Sequential(
            *[
                GRL(),
                nn.Linear(width, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 2)
            ])
        
        # initilization
        self.fc_layers[0].weight.data.normal_(0, 0.005)
        self.fc_layers[0].bias.data.fill_(0.1)
        self.classifier_layer[0].weight.data.normal_(0, 0.01)
        self.classifier_layer[0].bias.data.fill_(0.0)
        self.domain_discriminator[1].weight.data.normal_(0, 0.01)
        self.domain_discriminator[1].bias.data.fill_(0.0)
        self.domain_discriminator[3].weight.data.normal_(0, 0.01)
        self.domain_discriminator[3].bias.data.fill_(0.0)

    def forward(self, x):
        tmp = self.fc_layers(self.base_network(x))
        clf = self.classifier_layer(tmp)
        domain = self.domain_discriminator(tmp)
        return clf, domain


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)


class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

