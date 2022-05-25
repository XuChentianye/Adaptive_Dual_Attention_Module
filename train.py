from adamnet import *
from loss_fn import *
from indicators import *
from dataset import ISICDataset_Seg
from PIL import Image


import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchsummary import summary
import numpy as np
from torch.autograd import Variable


train_data_path = r'./ISIC-2017_Training_Data'
train_label_path = r'./ISIC-2017_Training_Part1_GroundTruth'

# val_data_path = r'./ISIC-2017_Validation_Data'
# val_label_path = r'./ISIC-2017_Validation_Part1_GroundTruth'

val_data_path = r'./ISIC-2017_Test_Data'
val_label_path = r'./ISIC-2017_Test_v2_Part1_GroundTruth'

train_dataset = ISICDataset_Seg(train_data_path, train_label_path)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

val_dataset = ISICDataset_Seg_Val(val_data_path, val_label_path)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"  # Choose device

# Using ResNet34 as the backbone network
model = ADAMNet_Seg(8, BasicBlock, [3, 4, 6, 3])



# load pretrained ResNet34 model
net_dict = model.state_dict()
snet_dict = model.sres.state_dict()
bnet_dict = model.bres.state_dict()
predict_model = torch.load('resnet34-333f7ec4.pth')
sstate_dict = {k: v for k, v in predict_model.items() if k in snet_dict.keys()}  # find shared layer parameters
# delete non-shared parameters
sstate_dict.pop('fc.weight')
sstate_dict.pop('fc.bias')
sstate_dict.pop('conv1.weight')

bstate_dict = {k: v for k, v in predict_model.items() if k in bnet_dict.keys()}  # find shared layer parameters
# delete non-shared parameters
bstate_dict.pop('fc.weight')
bstate_dict.pop('fc.bias')
bstate_dict.pop('conv1.weight')

# load pre-trained parameters into two ResNet34 in ADAMNet
snet_dict.update(sstate_dict)
bnet_dict.update(bstate_dict)
model.sres.load_state_dict(snet_dict)
model.bres.load_state_dict(bnet_dict)

model = model.to(device)

summary(model,(8, 512, 384))


# create a loss funcction
loss_fn = DiceCELoss()

# create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)



# define train function
def train(dataloader, model, loss_fn, optimizer, total_len, batch_size):
    model.train()
    total_batch = int(total_len/batch_size)
    loss, current, n = 0.0, 0.0, 0
    dsc, jsi, tji, se, sp, acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):
        n += 1
        # forward computing
        X, y = Variable(X).to(device), Variable(y).to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        
        # indicators
        cur_dsc = DSC(output, y)
        cur_jsi = JSI(output, y)
        cur_tji = TJI(output, y)
        cur_se = SE(output, y)
        cur_sp = SP(output, y)
        cur_acc = ACC(output, y)
        
        # back propagation
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        
        dsc += cur_dsc.item()
        jsi += cur_jsi.item()
        tji += cur_tji.item()
        se += cur_se.item()
        sp += cur_sp.item()
        acc += cur_acc.item()
        
        print("Proceed: "+ str(batch+1)+"/"+str(total_batch+1) + ", train_loss: " + str(np.round(cur_loss.item(),3))\
              + ", train_dsc: " + str(np.round(cur_dsc.item(),3))\
              + ", train_jsi: " + str(np.round(cur_jsi.item(),3))\
              + ", train_tji: " + str(np.round(cur_tji.item(),3))\
              + ", train_se: " + str(np.round(cur_se.item(),3))\
              + ", train_sp: " + str(np.round(cur_sp.item(),3))\
              + ", train_acc: " + str(np.round(cur_acc.item(),3)) , end="\r")
    print("\n\ntrain_loss: " + str(np.round(loss/n,3))\
          + "\ntrain_dsc: " + str(np.round(dsc/n,3))\
          + "\ntrain_jsi: " + str(np.round(jsi/n,3))\
          + "\ntrain_tji: " + str(np.round(tji/n,3))\
          + "\ntrain_se: " + str(np.round(se/n,3))\
          + "\ntrain_sp: " + str(np.round(sp/n,3))\
          + "\ntrain_acc: " + str(np.round(acc/n,3)), end="\r")
    return loss/n, acc/n

# define val function
def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    dsc, jsi, tji, se, sp, acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # forward computing
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            
            # indicators
            cur_dsc = DSC(output, y)
            cur_jsi = JSI(output, y)
            cur_tji = TJI(output, y)
            cur_se = SE(output, y)
            cur_sp = SP(output, y)
            cur_acc = ACC(output, y)
        
            loss += cur_loss.item()
            
            dsc += cur_dsc.item()
            jsi += cur_jsi.item()
            tji += cur_tji.item()
            se += cur_se.item()
            sp += cur_sp.item()
            acc += cur_acc.item()
        
            n += 1
        print("\n\ntest_loss: " + str(np.round(loss/n,3))\
              + "\ntest_dsc: " + str(np.round(dsc/n,3))\
              + "\ntest_jsi: " + str(np.round(jsi/n,3))\
              + "\ntest_tji: " + str(np.round(tji/n,3))\
              + "\ntest_se: " + str(np.round(se/n,3))\
              + "\ntest_sp: " + str(np.round(sp/n,3))\
              + "\ntest_acc: " + str(np.round(acc/n,3)), end="\r")
    return loss/n, acc/n


# Training
epoch = 250
min_acc = 0
train_loss, train_acc, test_loss, test_acc = [], [], [], []
for t in range(epoch):
    print(f'\n\n------------------ epoch {t+1} ------------------')
    tloss, tacc = train(train_dataloader, model, loss_fn, optimizer, train_dataset.__len__(), 32)
    train_loss.append(tloss)
    train_acc.append(tacc)
    if(t%10==9):
        vloss, vacc = val(val_dataloader, model, loss_fn)
        test_loss.append(vloss)
        test_acc.append(vacc)
    
#     # save model with best acc
#     if a > min_acc:
#         folder = 'save_model'
#         if not os.path.exists(folder):
#             os.mkdir('save_model')
#         min_acc = a
#         print('save best model')
#         torch.save(model.state_dict(), 'save_model/model.pth')

print('Done!')


print("train_loss:", train_loss)
print("test_loss:", test_loss)
print("train_acc:", train_acc)
print("test_acc:", test_acc)



# Test model: save sample predict
dataset = ISICDataset_Seg_Val(val_data_path, val_label_path)
dat = dataset[1666]

t_img = dat[0].to(device)
outp = model(t_img.reshape(1,8,512,384))
bpred = Image.fromarray(outp.detach().cpu().numpy()[0,0,:,:]>=0.5)
pred = Image.fromarray(outp.detach().cpu().numpy()[0,0,:,:])
bpred.save('bpred.tiff')
pred.save('pred.tiff')

t_mask = dat[1]
t_label = np.array(t_mask.detach().cpu().numpy())
t_label = Image.fromarray(t_label[0,:,:])
t_label.save('label.tiff')

t_image = Image.fromarray(t_img.detach().cpu().numpy()[0,:,:])
t_image.save('image.tiff')
