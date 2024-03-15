import torch
import torch.nn as nn

from src.utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Contants
        self.lambda_class =1
        self.lambda_noobj = 10
        self.lambda_obj =1
        self.lambda_box = 1

    def forward(self, preds, targets, anchors):
        obj = targets[...,0] == 1
        noobj = targets[..., 0] == 0

        # No Object Loss
        no_obj_loss = self.bce(
            (preds[..., 0:1][noobj]), (targets[...,0:1][noobj]))


        # Object Loss
        anchors = anchors.reshape(1,3,1,1,2)
        box_preds = torch.cat([self.sigmoid(preds[...,1:3]),
                                torch.exp(preds[...,3:5])*anchors], dim = -1)
        ious = intersection_over_union(box_preds[obj], targets[...,1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(preds[...,0:1][obj]), (ious * targets[...,0:1][obj]))

        # Box Coodinates Loss
        preds[..., 1:3] = self.sigmoid(preds[...,1:3]) # x, y to be between [0,1]
        targets[...,3:5] = torch.log(
            (1e-16 + targets[...,3:5]/anchors)
        )
        box_loss = self.mse(preds[...,1:5][obj], targets[...,1:5][obj])


        # Class Loss 
        calss_loss = self.entropy(
            (preds[...,5:][obj]), (targets[...,5][obj].long()),
        )


        return (
            self.lambda_box*box_loss +
            self.lambda_obj*object_loss +
            self.lambda_noobj*no_obj_loss +
            self.lambda_class*calss_loss
        )
