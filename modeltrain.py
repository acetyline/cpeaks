import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast as autocast
import copy
from typing import Optional, List
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from scipy.optimize import linear_sum_assignment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append('./cPeaks/deepCNN')
import DeepCNN
backbonemodel=torch.load('./cPeaks/deepCNN/epoch5_20230822.pth').to(device)

class loaddata(Dataset):
    def __init__(self,ls):
        self.ls=ls
    def __len__(self):
        return len(self.ls)
    def __getitem__(self,index):
        return self.ls[index]

batch_size=32

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
from typing import Callable
class DenseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        batch_norm: bool = True,
        dropout: float = 0.2,
        activation_fn: Callable = nn.GELU(),
    ):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features, bias=use_bias)
        self.batch_norm = (
            nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        )
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation_fn(x)
        return x
    
class decn1d(nn.Module):
    def __init__(self,num_queries=5):
        super(decn1d,self).__init__()
        self.stem=backbonemodel.stem
        self.tower=backbonemodel.tower
        self.pre_bottle=backbonemodel.pre_bottleneck
        self.bottle = DenseLayer(
            in_features=5120,
            out_features=160,
            use_bias=True,
            batch_norm=True,
            dropout=0.2,
            activation_fn=nn.Identity(),
        )
        hidden_dim=32
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.box_embed=MLP(hidden_dim,hidden_dim,2,3)#2表示左端位置和右端位置

    def forward(self,x):
        x=x.permute(0,2,1)
        x = self.stem(x)
        x = self.tower(x)
        x = self.pre_bottle(x)
        x = x.view(x.shape[0], -1)
        hs = self.bottle(x)
        hs=hs.view(32,5,32)
        hs=hs.unsqueeze(0)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.box_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

def midwtoxy(x):
    mid,w= x.unbind(-1)
    b = [(mid - 0.5 * w),(mid + 0.5 * w)]
    return torch.stack(b, dim=-1)

def xytomidw(a):
    x,y=a.unbind(-1)
    b=[((x+y)*0.5),(y-x)]
    return torch.stack(b,dim=-1)

def boxsize(x):
    return x[:,1]-x[:,0]

def box_iou(boxes1, boxes2):
    area1 = boxsize(boxes1)
    area2 = boxsize(boxes2)
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    union = area1[:, None] + area2 - wh
    iou = wh / union
    return iou, union

def getgiou(box1,box2):
    iou, union = box_iou(box1, box2)
    lt = torch.min(box1[:, None, 0], box2[:, 0])
    rb = torch.max(box1[:, None, 1], box2[:, 1])
    wh = (rb - lt).clamp(min=0)  
    return iou - (wh - union) / wh
    
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    #code from DETR model
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SetCriterion(nn.Module):
    #code from DETR model
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(midwtoxy(src_boxes), target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(getgiou(
            midwtoxy(src_boxes),
            target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    

def gettarget(x):
    result=[]
    #"labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
    #          objects in the target) containing the class labels
    #"boxes": Tensor of dim [num_target_boxes, 2] containing the target box coordinates
    for i in x:
        dic={}
        
        if i[0]<0:
            dic['boxes']=torch.tensor([[0.0,1.0]]).to(device)
            dic['labels']=[1]
        else:
            dic['boxes']=torch.tensor([[i[0],i[1]]]).to(device)
            dic['labels']=[0]
        for j in range(1,5):
            if i[j*2]<0:
                break 
            #print(torch.tensor([[i[j*2],i[j*2+1]]]).shape)
            #try:
            dic['boxes']=torch.cat((dic['boxes'],torch.tensor([[i[j*2],i[j*2+1]]]).to(device)),axis=0)
            #except:
                #print(dic['boxes'])
                #print(torch.tensor([i[j*2],i[j*2+1]]))
            dic['labels'].append(0)
        dic['labels']=torch.tensor(dic['labels']).to(device)
        result.append(dic)
    return result

class HungarianMatcher(nn.Module):
    #code from detr model
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(midwtoxy(out_bbox), tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -getgiou(midwtoxy(out_bbox), tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

matcher=HungarianMatcher()
weight_dict = {'loss_ce': 3, 'loss_bbox': 3}
weight_dict['loss_giou'] = 4
losses = ['labels', 'boxes', 'cardinality']
criterion=SetCriterion(num_classes=1,matcher=matcher,weight_dict=weight_dict,eos_coef=0.15,losses=losses)
criterion.to(device)
lr=0.0001
model=decn1d()
optimizer = optim.Adam(model.parameters(), lr=lr)
model=model.to(device)


trainpath='/data1/zyb/2/'
def train(model, optimizer, criterion, device):
    model.train()
    epoch=20
    i=0
    for epochs in range (epoch):
        for chrnum in range(1,23):
            try:
                del trainset
                del trainloader
            except:
                pass
            trainset=torch.load(trainpath+'chr'+str(chrnum)+'-train.pt')
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
            for data,target,_ in trainloader:
                if data.shape[0]!=32:
                    print(str(i)+'batchsize!=32')
                    continue
                i+=1
                data, target = data.to(device), target.to(device)
                target2=gettarget(target)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target2)
                weight_dict = criterion.weight_dict
                losses = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
                losses.backward()
                optimizer.step()
        torch.save(model,'/data1/zyb/model1/epochs'+str(epochs)+'.pt')

train(model,optimizer,criterion,device)
