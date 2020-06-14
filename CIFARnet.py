import warnings
warnings.simplefilter('ignore')
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

import torch
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import wandb

"""
1. Implement Conv and Fc blocks as pytorch modules
2. Implement CIFARnet as pytorch module with semi-automatic build
3. Add Pytorch-lightning maintenance
4. Tune Tensorboard logging
5. Run experiments
"""


class ConvMaxPoolBlock(nn.Module):
    def __init__(self, Cin: int, Cout: int, kernel_size: int, padding=0, stride=1, pool_kernel_size=(2,2), pool_stride=(2,2),
                 dropout_rate=0, BatchNorm=False, activation='relu', **kwargs):
        super(ConvMaxPoolBlock, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=kernel_size,
                                stride=stride, padding=padding, **kwargs)
        self.act = choose_activation(activation) if isinstance(activation, str) else activation
        self.BatchNorm2d = None
        self.dropout = None
        self.MaxPool2d = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        if BatchNorm:
            self.BatchNorm2d = nn.BatchNorm2d(num_features=Cout, momentum=0.1, affine=True)
        if dropout_rate != 0:
            assert 0 < dropout_rate <= 1, print('dropout_rate is a probability and should be in the interval 0<=p<=1')
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.act(x)
        if self.BatchNorm2d:
            x = self.BatchNorm2d(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.MaxPool2d(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, Cin: int, Cout: int, kernel_size: int, padding=0, stride=1,
                 dropout_rate=0, BatchNorm=False, activation='relu', **kwargs):
        super(ConvBlock, self).__init__()

        self.Conv2d = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=kernel_size,
                                stride=stride, padding=padding, **kwargs)
        self.act = choose_activation(activation) if isinstance(activation, str) else activation
        self.BatchNorm2d = None
        self.dropout = None
        if BatchNorm:
            self.BatchNorm2d = nn.BatchNorm2d(num_features=Cout, momentum=0.1, affine=True)
        if dropout_rate != 0:
            assert 0 < dropout_rate <= 1, print('dropout_rate is a probability and should be in the interval 0<=p<=1')
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.act(x)
        if self.BatchNorm2d:
            x = self.BatchNorm2d(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, Cin: int, Cout: int, kernel_size: int, padding=0, stride=1,
                 dropout_rate=0, BatchNorm=False, activation='relu', **kwargs):
        super(ConvBlock, self).__init__()

        self.Conv2d = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=kernel_size,
                                stride=stride, padding=padding, **kwargs)
        self.act = choose_activation(activation) if isinstance(activation, str) else activation
        self.BatchNorm2d = None
        self.dropout = None
        if BatchNorm:
            self.BatchNorm2d = nn.BatchNorm2d(num_features=Cout, momentum=0.1, affine=True)
        if dropout_rate != 0:
            assert 0 < dropout_rate <= 1, print('dropout_rate is a probability and should be in the interval 0<=p<=1')
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.act(x)
        if self.BatchNorm2d:
            x = self.BatchNorm2d(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class ResNetBottleneckBlock(nn.Module):
    def __init__(self, Cin: int, Cout: int, kernel_size: int, padding=0, stride=1, dropout_rate=0, activation='relu', **kwargs):
        super(ResNetBottleneckBlock, self).__init__()
        self.PseudoConv1 = nn.Conv2d(in_channels=Cin, out_channels=Cin, kernel_size=(1,1))
        self.Conv2d = nn.Conv2d(in_channels=Cin, out_channels=Cin, kernel_size=kernel_size,
                                 stride=stride, padding=padding, **kwargs)
        self.PseudoConv2 = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=(1,1))
        self.act = choose_activation(activation) if isinstance(activation, str) else activation
        self.BatchNorm2d = nn.BatchNorm2d(num_features=Cout, momentum=0.1, affine=True)
        self.shortcut = lambda x: x
        if dropout_rate != 0:
            assert 0 < dropout_rate <= 1, print('dropout_rate is a probability and should be in the interval 0<=p<=1')
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x0):
        x = self.PseudoConv1(x0)
        x = self.Conv2d(x)
        x = self.PseudoConv2(x)
        x = self.act(x)
        x = self.BatchNorm2d(x)
        if self.dropout:
            x = self.dropout(x)
        x += self.shortcut(x0)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, Cin: int, Cout: int, kernel_size: int, padding=1, stride=1,
                 dropout_rate=0, batchnorm2d=(True, True), activation='relu', shortcut_type='A', **kwargs):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=kernel_size,
                                stride=stride, padding=padding, **kwargs)
        self.conv2 = nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=kernel_size,
                                  stride=1, padding=padding, **kwargs)
        self.act1 = choose_activation(activation) if isinstance(activation, str) else activation
        self.act2 = choose_activation(activation) if isinstance(activation, str) else activation

        self.bn1 = nn.BatchNorm2d(num_features=Cout, momentum=0.1, affine=True) if batchnorm2d[0] else lambda x: x
        self.bn2 = nn.BatchNorm2d(num_features=Cout, momentum=0.1, affine=True) if batchnorm2d[1] else lambda x: x

        if shortcut_type == 'A':
            self.shortcut = lambda x: F.pad(x[..., ::stride, ::stride],
                                     pad=[0,0,0,0, (Cout-Cin)//2, (Cout-Cin)//2], mode="constant", value=0)
        elif shortcut_type == 'B':
            self.shortcut = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=(1,1), padding=0)
        self.dropout1 = None
        self.dropout2 = None
        if dropout_rate != 0:
            self.dropout1 = nn.Dropout2d(p=dropout_rate)
            self.dropout2 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x0):
        x = self.bn1(self.conv1(x0))
        if self.dropout1:
            x = self.dropout1(x)
        x = self.act1(x)
        x = self.bn2(self.conv2(x))
        if self.dropout2:
            x = self.dropout2(x)
        x += self.shortcut(x0)
        x = self.act2(x)
        return x

class FcBlock(nn.Module):
    def __init__(self, params: tuple, activations: tuple, dropouts: tuple, BatchNorm1ds: tuple, bias=True, xavier=True):
        super(FcBlock, self).__init__()
        self.layers = []
        self.activations = []
        self.dropouts = []
        self.batchnorms = []
        for i in range(len(params) - 1):
            L = nn.Linear(params[i], params[i + 1], bias=bias)
            if xavier: nn.init.xavier_uniform_(L.weight)
            A = choose_activation(activations[i])
            B = nn.BatchNorm1d(num_features=params[i+1]) if BatchNorm1ds[i] else lambda x: x
            D = nn.Dropout(p=dropouts[i]) if dropouts[i] else lambda x: x
            setattr(self, 'L' + str(i), L)
            setattr(self, 'B' + str(i), B)
            setattr(self, 'D' + str(i), D)
            self.layers.append(L)
            self.activations.append(A)
            self.batchnorms.append(B)
            self.dropouts.append(D)

    def forward(self, x):
        for L,A,B,D in zip(self.layers, self.activations, self.batchnorms, self.dropouts):
            x = D(B(A(L(x))))
        return x

class CIFARnetTemplate(pl.LightningModule):
    def __init__(self,):
        super(CIFARnetTemplate, self).__init__()
        self.hparams = None

    def forward(self, x):
        pass

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='/home/moonstrider/Neural_Networks_Experiments/data/', train=True,
                                                download=True, transform=transform)

        cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        cifar_train, cifar_val = random_split(trainset,  [45000, 5000]) #TODO: find a balanced version of random split and tune numbers

        # assign to use in dataloaders
        self.train_dataset = cifar_train
        self.val_dataset = cifar_val
        self.test_dataset = cifar_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=2)

    def loss_function(self, y, target):
        CE = F.cross_entropy(input=y, target=target, reduction='mean')
        return CE

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        labels_hat = torch.argmax(out, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        log = {'train_loss': loss, 'train_acc': acc}

        output = OrderedDict({'loss': loss,
                              'train_acc': acc,
                              'log': log
                              })
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        labels_hat = torch.argmax(out, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        miss_idx = torch.where(labels_hat != y)[0]
        log = {'val_loss': loss, 'val_acc': acc}

        y_hat_save = None if self.current_epoch < 12 else out.detach().cpu()
        output = OrderedDict({'val_loss': loss,
                              'y_hat': y_hat_save,
                              'val_acc': acc,
                              'missclass': (x[miss_idx], labels_hat[miss_idx], y[miss_idx]),
                              'log': log
                              })
        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        labels_hat = torch.argmax(out, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        miss_idx = torch.where(labels_hat != y)[0]

        log = {'test_loss': loss, 'test_acc': acc}
        output = OrderedDict({'test_loss': loss,
                              'y_hat': out,
                              'y': y,
                              'test_acc': torch.tensor(acc),  # everything must be a tensor
                              'missclass': (x[miss_idx], labels_hat[miss_idx], y[miss_idx]),
                              'log': log
                              })
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'],
                                     weight_decay=self.hparams['weight_decay'])
        return optimizer

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()

        mean_acc = 0
        for output in outputs:
            mean_acc += output['train_acc']
        mean_acc /= len(outputs)

        tqdm_dict = {'avg_train_loss': avg_loss, 'train_acc': mean_acc}
        log = {'avg_train_loss': avg_loss, 'train_accuracy': mean_acc}
        results = {
            'val_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': log
        }
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        missed_print = torch.cat([x['missclass'][0] for x in outputs[:4]], dim=0)

        mean_acc = 0
        for output in outputs:
            mean_acc += output['val_acc']
        mean_acc /= len(outputs)

        tqdm_dict = {'avg_val_loss': avg_loss, 'val_acc': mean_acc}
        self._log_images(missed_print, name='missclassified_images_val')

        log = {'avg_val_loss': avg_loss, 'val_accuracy': mean_acc}
        results = {
            'val_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': log
        }
        return results

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().item()
        y_hats = torch.cat([x['y_hat'] for x in outputs])
        ys = torch.cat([x['y'] for x in outputs])
        missed_print = torch.cat([x['missclass'][0] for x in outputs[:4]])

        mean_acc = 0
        for output in outputs:
            mean_acc += output['test_acc']
        mean_acc /= len(outputs)

        self._log_images(missed_print, name='missclassified_images_test')
        log = {'avg_test_loss': avg_loss, 'test_accuracy': mean_acc}
        results = {
            'test_predictions': y_hats,
            'test_targets': ys,
            'avg_test_loss': avg_loss,
            'log': log
        }
        return results

    def _log_images(self, images, name, k=4):
        if 'WandbLogger' in self.logger.__str__():
            self.logger.experiment.log({name: [wandb.Image(i) for i in images[:k]]})
        else:
            grid = torchvision.utils.make_grid(images[:k], nrow=4)
            self.logger.experiment.add_image(name, grid, 0)

class CIFARnet(CIFARnetTemplate):
    def __init__(self, hparams):
        super(CIFARnet, self).__init__()
        self.conv_blocks = []
        self.hparams = hparams
        for conv_i, conv_type in zip(hparams['Conv2dBlocks'], hparams['ConvBlocksTypes']):
            if conv_type == 'MaxPool':
                block = ConvMaxPoolBlock(**hparams['Conv2dBlocks'][conv_i])
            else:
                block = ConvBlock(**hparams['Conv2dBlocks'][conv_i])
            setattr(self, 'Conv2dBlock_' + str(conv_i), block)
            self.conv_blocks.append(block)
        self.FcBlock = FcBlock(**hparams['FcBlock'])

    def forward(self, x):
        for Block in self.conv_blocks:
            x = Block(x)
        x = x.view(x.size(0), -1)
        x = self.FcBlock(x)
        return x

class ResNetCifar(CIFARnetTemplate):
    def __init__(self, hparams):
        super(ResNetCifar, self).__init__()
        self.hparams = hparams
        self.CvBundles = []
        if hparams['FirstMaxPool']:
            self.FirstConv = ConvMaxPoolBlock(**hparams['FirstConvBlock'])
        else:
            self.FirstConv = ConvBlock(**hparams['FirstConvBlock'])
        for bundle in hparams['Conv2dBundles']:
            tuft = self._weave_tuft(**hparams['Conv2dBundles'][bundle])
            setattr(self, bundle, tuft)
            self.CvBundles.append(tuft)
        self.FcBlock = nn.Linear(*hparams['FcBlock'])
        self.AvgPool2d = nn.AvgPool2d(kernel_size=(hparams['out_of_conv_dim']))

        self.apply(_weights_init)

    def _weave_tuft(self, blocks_num, Cin_first, stride_first, block_hparams):
        bundle = []
        block_I_hparams = block_hparams.copy()
        block_I_hparams['Cin'] = Cin_first
        block_I_hparams['stride'] = stride_first
        bundle.append(ResNetBlock(**block_I_hparams))
        for _ in range(1, blocks_num):
            bundle.append(ResNetBlock(**block_hparams))
        return nn.Sequential(*bundle)

    def forward(self, x):
        x = self.FirstConv(x)
        for bundle in self.CvBundles:
            x = bundle(x)
        x = self.AvgPool2d(x)
        x = x.view(x.size(0), -1)
        x = self.FcBlock(x)
        return x

class MyNet(CIFARnetTemplate):
    def __init__(self, hparams):
        super(MyNet, self).__init__()
        self.hparams = hparams
        self.CvBundles = []
        if hparams['FirstMaxPool']:
            self.FirstConv = ConvMaxPoolBlock(**hparams['FirstConvBlock'])
        else:
            self.FirstConv = ConvBlock(**hparams['FirstConvBlock'])
        for bundle in hparams['Conv2dBundles']:
            tuft = self._weave_tuft(**hparams['Conv2dBundles'][bundle])
            setattr(self, bundle, tuft)
            self.CvBundles.append(tuft)
        self.FcBlock = nn.Linear(*hparams['FcBlock'])

        self.apply(_weights_init)

    def _weave_tuft(self, blocks_num, Cin_first, stride_first, block_hparams):
        bundle = []
        block_I_hparams = block_hparams.copy()
        block_I_hparams['Cin'] = Cin_first
        block_I_hparams['stride'] = stride_first
        bundle.append(ResNetBlock(**block_I_hparams))
        for _ in range(1, blocks_num):
            bundle.append(ResNetBlock(**block_hparams))
        return nn.Sequential(*bundle)

    def forward(self, x):
        x = self.FirstConv(x)
        for bundle in self.CvBundles:
            x = bundle(x)
        x = self.AvgPool2d(x)
        x = x.view(x.size(0), -1)
        x = self.FcBlock(x)
        return x


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def choose_activation(name):
    if name == 'relu':
        return F.relu
    elif name == 'leaky_relu':
        return F.leaky_relu
    elif name == 'sigmoid':
        return F.sigmoid
    elif name == 'tanh':
        return F.tanh
    elif name == 'None':
        return lambda x: x
    else:
        raise ValueError(f'Function {name} is not available as an activation function for this model.')

def compute_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp