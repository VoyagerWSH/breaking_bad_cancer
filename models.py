import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from src.cindex import concordance_index
from math import floor
import random
import wandb

from torchvision.models import resnet18

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=9, init_lr=1e-4, optimizer="Adam", loss="Cross Entropy"):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes
        self.optimizer = optimizer

        # define loss
        if loss == "Cross Entropy":
            self.loss = nn.CrossEntropyLoss(label_smoothing=0.005)
        if loss == "Binary Cross Entropy":
            self.loss = nn.BCEWithLogitsLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)

        # store pred
        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def get_xy(self, batch):
        if isinstance(batch, list):
            x, y = batch[0], batch[1]
        else:
            assert isinstance(batch, dict)
            x, y = batch["x"], batch["y_seq"][:,0]
        return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)
        
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.init_lr)
        elif self.optimizer == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=self.init_lr)
        elif self.optimizer == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.init_lr)




class MLP(Classifer):
    def __init__(self, in_features=28*28*3, num_classes = 9, n_fc=2, hidden_dim=1024, use_bn=True, init_lr = 1e-3, dropout_p=0, optimizer = "Adam", loss = "Cross Entropy",**kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()
        
        self.fc_layers = nn.ModuleList()
        self.use_bn = use_bn

        out_features = hidden_dim
        for i in range(n_fc):
            self.fc_layers.append(nn.Linear(in_features, out_features))
            if self.use_bn:
                self.fc_layers.append(nn.BatchNorm1d(out_features))

            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_p))
            in_features = out_features
            
        self.fc_layers.append(nn.Linear(out_features, num_classes))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, channels*width*height)

        for layer in self.hidden_layers:
            x = layer(x)

        return x

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(dim, kernel_size=1, stride=1, padding=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    dim = floor( ((dim + (2 * padding) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    return dim

class CNN(Classifer):
    def __init__(self, conv_layers=[], in_dim = 28, num_classes = 9, pooling=None, use_bn=True, init_lr = 1e-3, optimizer = "Adam", loss = "Cross Entropy",**kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()
        
        self.conv_layers = nn.ModuleList()
        self.use_bn = use_bn
        self.dim = in_dim
        self.num_classes = num_classes
        
        # conv -> norm -> relu -> pool
        for i in range(len(conv_layers)-2):
            self.conv_layers.append(nn.Conv2d(in_channels=conv_layers[i], out_channels=conv_layers[i+1], kernel_size=3, stride=1, padding=0, padding_mode='zeros'))
            self.dim = conv_output_shape(self.dim, kernel_size=3, stride=1, padding=0)
            
            if use_bn:
                self.conv_layers.append(nn.BatchNorm2d(conv_layers[i+1], eps=1e-5, momentum=0.1))
            
            self.conv_layers.append(nn.ReLU())

            if pooling == "max":
                self.conv_layers.append(nn.MaxPool2d(3, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=3, stride=2)
            if pooling == "avg":
                self.conv_layers.append(nn.AvgPool2d(3, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=3, stride=2)
        
        self.conv_layers.append(nn.Conv2d(in_channels=conv_layers[-2], out_channels=conv_layers[-1], kernel_size=3, stride=1, padding=0, padding_mode='zeros'))
        self.dim = conv_output_shape(self.dim, kernel_size=3, stride=1, padding=0)
        
        # global pooling to obtain C*1*1 image
        self.conv_layers.append(nn.MaxPool2d(self.dim))
        self.conv_layers.append(nn.Linear(conv_layers[-1], self.num_classes))
    
    def forward(self, x):
        for layer in self.conv_layers[:-1]:
            x = layer(x)
        return self.conv_layers[-1](x.flatten(1))

class Resnet(Classifer):
    def __init__(self, num_classes = 9, use_bn=True, init_lr = 1e-3, optimizer = "Adam", loss = "Cross Entropy", pre_train = True, dropout_p=0, n_fc = 2, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()

        self.use_bn = use_bn
        self.num_classes = num_classes
        self.fc_layers = nn.ModuleList()
        self.pre_train = pre_train

        if pre_train:
            self.backbone = resnet18(weights="DEFAULT")
        else:
            self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.out_features
        out_features = 512

        self.fc_layers.append (nn.ReLU())
        for i in range(n_fc):
            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_layers.append (nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_p))
            in_features = out_features
        self.fc_layers.append(nn.Linear(out_features, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        for layer in self.fc_layers:
            x = layer(x)

        return x


class CNN_3D(Classifer):
    def __init__(self, conv_layers=[], in_dim = 256, in_depth = 200, num_classes = 2, pooling=None, use_bn=True, init_lr = 1e-3, optimizer = "Adam", loss = "Cross Entropy",**kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()
        
        self.conv_layers = nn.ModuleList()
        self.use_bn = use_bn
        self.dim = in_dim
        self.depth = in_depth
        self.num_classes = num_classes
        
        # conv -> norm -> relu -> pool
        for i in range(len(conv_layers)-2):
            self.conv_layers.append(nn.Conv3d(in_channels=conv_layers[i], out_channels=conv_layers[i+1], kernel_size=5, stride=1, padding=0, padding_mode='zeros'))
            self.dim = conv_output_shape(self.dim, kernel_size=5, stride=1, padding=0)
            self.depth = conv_output_shape(self.depth, kernel_size=5, stride=1, padding=0)
            
            if use_bn:
                self.conv_layers.append(nn.BatchNorm3d(conv_layers[i+1], eps=1e-5, momentum=0.1))
            
            self.conv_layers.append(nn.ReLU())

            if pooling == "max":
                self.conv_layers.append(nn.MaxPool3d(5, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=5, stride=2)
                self.depth = conv_output_shape(self.depth, kernel_size=5, stride=2)
            if pooling == "avg":
                self.conv_layers.append(nn.AvgPool3d(5, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=5, stride=2)
                self.depth = conv_output_shape(self.depth, kernel_size=5, stride=2)
        
        self.conv_layers.append(nn.Conv3d(in_channels=conv_layers[-2], out_channels=conv_layers[-1], kernel_size=5, stride=2, padding=0, padding_mode='zeros'))
        self.dim = conv_output_shape(self.dim, kernel_size=5, stride=2, padding=0)
        self.depth = conv_output_shape(self.depth, kernel_size=5, stride=2, padding=0)
    
        # global pooling to obtain C*1*1*1 image
        self.conv_layers.append(nn.MaxPool3d((self.depth, self.dim, self.dim)))
        self.conv_layers.append(nn.Linear(conv_layers[-1], self.num_classes))

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[4], x.shape[2], x.shape[3]))
        for layer in self.conv_layers[:-1]:
            x = layer(x)
        x = x.flatten(1)
        x = self.conv_layers[-1](x)
        return x
    
class Resnet_2D_to_3D(Classifer):
    def __init__(self, num_classes = 2, init_lr = 1e-3, optimizer = "AdamW", loss = "Cross Entropy", pre_train = True, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.fc_layers = nn.ModuleList()
        self.pre_train = pre_train

        if pre_train:
            self.backbone = resnet18(weights="DEFAULT")
        else:
            self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.out_features
        self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(in_features, 512))
        self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(512, num_classes))
        self.fc_layers.append(nn.ReLU())
        self.final_layer = nn.Linear(67 * 2, 2)

    def forward(self, x):

        # (BCHWD -> BCDHW) for conv_3d
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[4], x.shape[2], x.shape[3]))
        # (BCDHW -> BDHW) after squeeze
        x = x.squeeze(1)
        # duplicate channel values to fit in ResNet
        x = torch.cat((x, x[:,0:1,:,:]),1)
        lst = []
        for i in range(0, x.shape[1], 3):
            x_sub = x[:,i:i+3,:,:]
            x_sub = self.backbone(x_sub)
            for layer in self.fc_layers:
                x_sub = layer(x_sub)
            lst.append(x_sub)
        x = torch.cat(tuple(lst),1)
        x = self.final_layer(x)
        return x

class Resnet_3D(Classifer):
    def __init__(self, num_classes = 2, init_lr = 1e-3, optimizer = "AdamW", loss = "Cross Entropy", pre_train = True, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.fc_layers = nn.ModuleList()

        if pre_train:
            self.backbone = torch.load('checkpoints/r3d_18.pt')
            # self.backbone = torchvision.models.video.r3d_18(weights="DEFAULT")
        else:
            self.backbone = torchvision.models.video.r3d_18()

        # change global avg pool to global max pool
        self.backbone.avgpool = nn.AdaptiveMaxPool3d(1)

        # average over the conv_3d filter channels to fit the input of channel 1
        sd = self.backbone.state_dict()
        conv_c1 = torch.mean(sd['stem.0.weight'], dim=1).unsqueeze(1)
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        sd['stem.0.weight'] = conv_c1
        self.backbone.load_state_dict(sd)

        # change the last fc layer to fit the number of classes
        self.backbone.fc = nn.Linear(512, 128)
        self.fc_layers.append (nn.ReLU())
        self.fc_layers.append(nn.Linear(128, self.num_classes))

    def forward(self, x):
        # (BCHWD -> BCDHW) for conv_3d
        x = torch.permute(x, (0, 1, 4, 2, 3))
        
        x = self.backbone(x)
        for layer in self.fc_layers:
            x = layer(x)

        return x

class Attn_Guided_Resnet(Classifer):
    def __init__(self, num_classes = 2, init_lr = 1e-3, optimizer = "AdamW", loss = "Cross Entropy", pre_train = True, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.layers_after_attn = nn.ModuleList()

        if pre_train:
            # self.backbone = torch.load('checkpoints/r3d_18.pt')
            self.backbone = torchvision.models.video.r3d_18(weights="DEFAULT")
        else:
            self.backbone = torchvision.models.video.r3d_18()

        # At the strat of ResNet: average over the conv_3d filter channels to fit the input of channel 1
        sd = self.backbone.state_dict()
        conv_c1 = torch.mean(sd['stem.0.weight'], dim=1).unsqueeze(1)
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        sd['stem.0.weight'] = conv_c1
        self.backbone.load_state_dict(sd)

        # delete the avgpool layer and the last fc layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # attention convolution (weighted avg pool)
        # out_channel = 1 because we want to collapse the attention of all channels into one
        self.attn_pool = nn.Conv3d(512, 1, kernel_size=1, stride=1)

        # change the last fc layer to fit the number of classes
        self.layers_after_attn.append(nn.Linear(512, 128))
        self.layers_after_attn.append(nn.BatchNorm1d(128))
        self.layers_after_attn.append(nn.ReLU())
        self.layers_after_attn.append(nn.Linear(128, self.num_classes))

    def get_xy(self, batch):
        assert isinstance(batch, dict)
        x, y, mask = batch["x"], batch["y_seq"][:,0], batch["mask"]
        return x, y.to(torch.long).view(-1), mask
    
    def attn_guided_loss(self, attn_map, mask, visualize=False):
        # downsample the mask to the embedding space of the attention map
        self.adpt_max_pool = nn.AdaptiveMaxPool3d(attn_map.shape[2:])
        downsampled_mask = self.adpt_max_pool(mask)
        
        # a true false list indicating the batch index with annotation
        batch_idx_with_annotation = torch.sum(downsampled_mask, dim=(1,2,3,4)) > 0
        
        # upsample the attention map to a mask when needed (i.e., val & test)
        if visualize:
            self.upsampler = nn.Upsample(size=mask.shape[2:])
            attn_mask = self.upsampler(attn_map)
            
            # return loss = 0 if no bounding box is drawn
            if sum(batch_idx_with_annotation) == 0:
                return 0, attn_mask
            
            # compute the loss otherwise
            attn_loss = -torch.log(torch.dot(downsampled_mask[batch_idx_with_annotation].view(-1), attn_map[batch_idx_with_annotation].view(-1))+1e-8)
            return attn_loss/sum(batch_idx_with_annotation), attn_mask
        
        # no visualization for train set
        if sum(batch_idx_with_annotation) == 0:
            return 0
        
        attn_loss = -torch.log(torch.dot(downsampled_mask[batch_idx_with_annotation].view(-1), attn_map[batch_idx_with_annotation].view(-1))+1e-8)    
        return attn_loss/sum(batch_idx_with_annotation)
    
    def training_step(self, batch, batch_idx):
        x, y, mask = self.get_xy(batch)
        # (BCHWD -> BCDHW) for conv_3d
        x = torch.permute(x, (0, 1, 4, 2, 3))
        mask = torch.permute(mask, (0, 1, 4, 2, 3))

        y_hat, attn_map = self.forward(x)
        pred_loss = self.loss(y_hat, y)
        attn_loss = self.attn_guided_loss(attn_map, mask)
        loss = pred_loss + attn_loss

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = self.get_xy(batch)
        # (BCHWD -> BCDHW) for conv_3d
        x = torch.permute(x, (0, 1, 4, 2, 3))
        mask = torch.permute(mask, (0, 1, 4, 2, 3))

        y_hat, attn_map = self.forward(x)
        pred_loss = self.loss(y_hat, y)
        attn_loss, attn_mask = self.attn_guided_loss(attn_map, mask, visualize=True)
        loss = pred_loss + attn_loss

        if y.sum() > 0:
            print("Annotation here")

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        
        # # visualizing each sample in the batch
        # x = ((x * 87.1849) + 128.1722).int()
        # # softmax!!
        # class_labels = {0: "non-cancer", 1: "cancer"}

        # class_set = wandb.Classes(
        #     [
        #         {"name": "non-cancer", "id": 0},
        #         {"name": "cancer", "id": 1}
        #     ]
        # )
        # for i in range(len(x)):
        #     self.log(
        #     {"For val batch #" + str(batch_idx) + " img #" + str(i): 
        #      wandb.Image(x[i], 
        #                 masks={
        #                      "predictions": {"mask_data" : attn_mask[i], "class_labels" : class_labels},
        #                      "ground_truth": {"mask_data" : mask[i], "class_labels" : class_labels}
        #                 },
        #                 classes=class_set
        #     )})

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = self.get_xy(batch)
        # (BCHWD -> BCDHW) for conv_3d
        x = torch.permute(x, (0, 1, 4, 2, 3))
        mask = torch.permute(mask, (0, 1, 4, 2, 3))

        y_hat, attn_map = self.forward(x)
        pred_loss = self.loss(y_hat, y)
        attn_loss, attn_mask = self.attn_guided_loss(attn_map, mask, visualize=True)
        loss = pred_loss + attn_loss

        # self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        # self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        
        # # visualizing each sample in the batch
        # for i in range(len(x)):
        #     self.log(
        #     {"For test batch #" + str(batch_idx) + " img #" + str(i): wandb.Image(x[i], masks={
        #         "predictions" : {
        #             "mask_data" : attn_mask[i],
        #             "class_labels" : y_hat[i]
        #         },
        #         "ground_truth" : {
        #             "mask_data" : mask[i],
        #             "class_labels" : y[i]
        #         }
        #     })})

        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def forward(self, x):
        # run the model to get the embedding
        z = self.backbone(x)

        # compute alpha for attantion guided pooling
        alpha = self.attn_pool(z)
        B, C_, D_, H_, W_ = alpha.shape
        assert(C_ == 1)
        alpha = F.softmax(alpha.view(B, -1), dim=1).view(B, C_, D_, H_, W_)

        # attention guided pooling
        output = (alpha * z).sum(dim=(2, 3, 4))
        
        for layer in self.layers_after_attn:
            output = layer(output)

        return output, alpha
