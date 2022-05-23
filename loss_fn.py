import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, pred, label):
		batch_size = label.size()[0]  # fetch batch_size
		smooth = 100  # smooth variable (in case of denominator = 0)

		pred_flat = pred.view(batch_size, -1)
		label_flat = label.view(batch_size, -1)

		intersection = pred_flat * label_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + label_flat.sum(1) + smooth)

		loss = 1 - N_dice_eff.sum() / batch_size  # the average loss of this batch
		return loss

# class BinaryCrossEntropyLoss(nn.Module):
# 	def __init__(self):
# 		super(BinaryCrossEntropyLoss, self).__init__()
	
# 	def forward(self, pred, label):
# 		batch_size = label.size()[0]  # fetch batch_size

# 		pred_flat = pred.view(batch_size, -1)+0.11
# 		label_flat = label.view(batch_size, -1)
# 		one_minus_pred = 1 - pred_flat
# 		one_minus_label = 1 - label_flat
# 		log_pred = torch.log(pred_flat)
# 		log_one_minus_pred = torch.log(one_minus_pred)

# 		#loss_combine = - (label_flat * log_pred + one_minus_label * log_one_minus_pred)/200000
# 		loss_combine = - (label_flat * log_pred)/200000
# 		loss = loss_combine.sum() / batch_size  # the average loss of this batch
# 		print(loss)
# 		return loss

# L_DiceCE = L_CE + L_Dice
class DiceCELoss(nn.Module):
	def __init__(self):
		super(DiceCELoss, self).__init__()
		self.dice = BinaryDiceLoss()
		# self.bce = BinaryCrossEntropyLoss()
		self.bce = nn.BCELoss()
	
	def forward(self, pred, label):
		loss = self.dice(pred, label) + self.bce(pred, label)
		return loss

