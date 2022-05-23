import torch
import torch.nn as nn

def DSC(pred, label):
	batch_size = label.size()[0]  # fetch batch_size
	smooth = 0.01  # smooth variable (in case of denominator = 0)

	pred_flat = pred.view(batch_size, -1)
	label_flat = label.view(batch_size, -1)
	
	intersection = pred_flat * label_flat 
	N_dice = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + label_flat.sum(1) + smooth)
	
	out = N_dice.sum() / batch_size
	return out

def JSI(pred, label):
    batch_size = label.size()[0]  # fetch batch_size

    pred_flat = pred.view(batch_size, -1)
    label_flat = label.view(batch_size, -1)
	
    intersection = pred_flat * label_flat
    N_jsi = intersection.sum(1) / (pred_flat.sum(1) + label_flat.sum(1) - intersection.sum(1))
	
    out = N_jsi.sum() / batch_size
    return out

def TJI(pred, label):
    batch_size = label.size()[0]  # fetch batch_size

    pred_flat = pred.view(batch_size, -1)
    label_flat = label.view(batch_size, -1)
	
    intersection = pred_flat * label_flat
    N_jsi = intersection.sum(1) / (pred_flat.sum(1) + label_flat.sum(1) - intersection.sum(1))
    mask = N_jsi >= 0.65
    N_jsi = N_jsi * mask
    out = N_jsi.sum() / batch_size
    return out

def SE(pred, label):
    batch_size = label.size()[0]  # fetch batch_size
    pred_flat = pred.view(batch_size, -1)
    label_flat = label.view(batch_size, -1)

    pred_flat = pred_flat >= 0.5
    tp_flat = pred_flat * label_flat

    N_se = tp_flat.sum(1) / label_flat.sum(1)
    out = N_se.sum() / batch_size
    return out

def SP(pred, label):
    batch_size = label.size()[0]  # fetch batch_size
    pred_flat = pred.view(batch_size, -1)
    label_flat = label.view(batch_size, -1)

    neg_pred_flat = pred_flat < 0.5
    neg_label_flat = label_flat < 0.5

    tn_flat = neg_pred_flat * neg_label_flat

    N_sp = tn_flat.sum(1) / neg_label_flat.sum(1)
    out = N_sp.sum() / batch_size
    return out

def ACC(pred, label):
    batch_size = label.size()[0]  # fetch batch_size
    pred_flat = pred.view(batch_size, -1)
    label_flat = label.view(batch_size, -1)

    neg_pred_flat = pred_flat < 0.5
    neg_label_flat = label_flat < 0.5

    tp_flat = pred_flat * label_flat
    tn_flat = neg_pred_flat * neg_label_flat

    N_acc = (tp_flat.sum(1) + tn_flat.sum(1)) / (label_flat.sum(1) + neg_label_flat.sum(1))
    out = N_acc.sum() / batch_size
    return out