#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Check nvcc version
get_ipython().system('nvcc -V')
# Check GCC version
get_ipython().system('gcc --version')


# In[2]:


# For Google Colaboratory
import sys, os
if 'google.colab' in sys.modules:
    # mount google drive
    from google.colab import drive
    drive.mount('/content/gdrive')
 
    path_to_file = '/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2'
    print(path_to_file)
    # change current path to the folder containing "file_name"
    os.chdir(path_to_file)
    get_ipython().system('pwd')


# In[3]:


get_ipython().run_line_magic('cd', '../../../../../../')
get_ipython().system('ls')


# In[4]:


get_ipython().system('git clone https://github.com/xinntao/BasicSR.git')
get_ipython().run_line_magic('cd', 'BasicSR')
get_ipython().system('pip install -r requirements.txt')
get_ipython().system('python3 setup.py develop')
get_ipython().system('python setup.py develop --no_cuda_ext')


# In[1]:


import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from PIL import Image 
import matplotlib.pyplot as plt
device=torch.device('cuda')
get_ipython().system('pwd')
from basicsr.models.archs.srresnet_arch_d import MSRResNet


# In[2]:



model = MSRResNet(
    num_in_ch=3, num_out_ch=3).to(device)

model.to(device)


# In[3]:


sum(p.numel() for p in model.parameters())


# In[4]:


import math
import cv2
def compute_psnr(img1, img2):
  img1 = img1.astype(np.float64) / 255.
  img2 = img2.astype(np.float64) / 255.
  mse = np.mean((img1 - img2) ** 2)
  if mse == 0:
      return "Same Image"
  return 10 * math.log10(1. / mse)


# In[5]:


train_dir = "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/data/Train/LR_x4"
train_target = "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/data/Train/HR"



####################################################
val_dir = "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/data/Val/LR_x4"
val_target = "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/data/Val/HR"
bs = 1
workers = 0

def train_preprocess(datadir):
    test_transforms = transforms.Compose([
        #can try random crop with at size of the shorter edge
        #transforms.RandomVerticalFlip(p=0.1),
        transforms.ToTensor() 
        ]) 

    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
 
    testloader = torch.utils.data.DataLoader(test_data,
                    batch_size=bs,shuffle=False,    # use custom collate function here
                       num_workers=workers)
    return testloader
 

def preprocess(datadir):
    test_transforms = transforms.Compose([
        #can try random crop with at size of the shorter edge
        #transforms.RandomCrop(16),
        transforms.ToTensor() 
        ]) 

    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
 
    testloader = torch.utils.data.DataLoader(test_data,
                    batch_size=bs,shuffle=False,    # use custom collate function here
                       num_workers=workers)
    return testloader
 
trainloader = train_preprocess(train_dir)
targetloader = preprocess(train_target)
val_loader =  preprocess(val_dir)
val_targetloader = preprocess(val_target)


# In[6]:


def val_stage1(val_loader, val_targetloader):

  total_psnr  = 0 
 
  for i, (img, target) in enumerate(zip(val_loader , val_targetloader)):
    #if i==0:
      model.eval()  
      with torch.no_grad():    
        input, target_data = img[0].to(device), target[0].to(device)
        output=model(input)
        #output=G(output)


        output_array = output[0].detach().cpu().permute(1,2,0).numpy() *255
        target_array = target_data[0].detach().cpu().permute(1,2,0).numpy() *255
        psnr_base = compute_psnr(output_array, target_array)
        if i% 20==0:
          print(i,'PSNR for validation image: ', psnr_base)
 
        total_psnr  = total_psnr  +  psnr_base
    

  ave_psnr = total_psnr/(i+1)
  print('Average PSNR for image: ', ave_psnr)
  return ave_psnr

  


# In[9]:


lr = 0.0002
beta1 = 0.5
from torch import nn, optim
mse_criterion  = nn.MSELoss()
l1_criterion  = nn.L1Loss()
model = MSRResNet(
    num_in_ch=3, num_out_ch=3).to(device)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
import numpy as np
import time
import os
from torchvision.utils import save_image
num_epochs = 100
 
psnr_train_epoch  = []
psnr_valid_epoch  = []
running_loss = []
for epoch in range(num_epochs):

  psnr_ratio=0
  total_loss = 0
  start = time.time()
  for i, (img, target) in enumerate(zip(trainloader, targetloader)):
    #if i==0: 
      input, target_data = img[0].to(device), target[0].to(device)
      output=model(input)
      model.zero_grad() 
      mse_loss = mse_criterion(output, target_data)
      l1_loss =  l1_criterion(output, target_data)
 
      mse_loss.backward(retain_graph=True)
      l1_loss.backward(retain_graph=True)
      LOSS = (mse_loss.detach().item() + l1_loss.detach().item()) 
      total_loss = total_loss + LOSS
      optimizer.step()   
      model.zero_grad()

      output_array = output[0].detach().cpu().permute(1,2,0).numpy() *255
      target_array = target_data[0].detach().cpu().permute(1,2,0).numpy() *255
      psnr_base = compute_psnr(output_array, target_array)
      psnr_ratio=psnr_ratio+ psnr_base
         
      if i%100 ==0:
          print(i, LOSS,'PSNR for image: ', psnr_base)

      ############################plot#################
      if psnr_base<0:
        print(i, LOSS,'PSNR for image: ', psnr_base)
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.title.set_text('Output')
        ax1.axis('off')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.title.set_text('Ground Truth')
        ax2.axis('off')
        
        
        out_img = np.clip(output[0].detach().cpu().permute(1,2,0).numpy(), 0, 1)
        tgt_img = np.clip(target_data[0].detach().cpu().permute(1,2,0).numpy(), 0, 1)
        ax1.imshow(out_img)
        ax2.imshow(tgt_img)
      ######################################################
  ave_loss = total_loss/(i+1)
  running_loss.append(ave_loss) 

  ave_psnr = psnr_ratio/(i+1)
  val_psnr = val_stage1(val_loader, val_targetloader)

  psnr_train_epoch.append(ave_psnr)
  psnr_valid_epoch.append(val_psnr)

  
  stop = time.time()
  duration = stop-start
  print(epoch, ave_psnr, ave_loss, 'time: ', duration/60)
  if epoch>0 and psnr_valid_epoch[epoch] <psnr_valid_epoch[epoch-1]:
    torch.save(model, "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/results/test20_mse_Loss.pth")
    break
  if epoch==num_epochs-1:
    torch.save(model, "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/results/test20_mse_Loss.pth")

plt.figure(figsize=(5,5))
plt.title("Train PSNR vs. Test PSNR")
plt.plot(psnr_train_epoch,label="Training") 
plt.plot(psnr_valid_epoch,label="Validation") 
plt.xlabel("iterations")
plt.ylabel("PSNR")
plt.legend()
    
plt.savefig('test20_mse_Loss', dpi=300, bbox_inches='tight')
plt.show()
        
  
      ########### process directly from tensor also can ########################
      #model(img_tensor_data)


# In[11]:


###############for submission #########################################
from torchvision.utils import save_image
import glob
import os
source_dir =  "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/data/Test/LR/LR/"
test_dir = "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/data/Test/LR/"
save_path =  "/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/data/Test/HR/new"
model = MSRResNet(
    num_in_ch=3, num_out_ch=3).to(device)
print(sum(p.numel() for p in model.parameters()))
stage1_model = torch.load("/content/gdrive/My Drive/AI6126 Advanced CV Assignment/Project 2/results/test20_mse_Loss.pth")
testloader = preprocess(test_dir)
 
def inference(testloader):
  for i, (img, f) in enumerate(zip(testloader,sorted(os.listdir(source_dir)))):
        img_name = f
        input  = img[0].to(device) 
        output=stage1_model(input)
        #output=G(output)
        save_image(output[0], os.path.join(save_path, img_name))
        if i %20==0:
          print(img_name, 'saved!')
                  
 

  
inference(testloader)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




