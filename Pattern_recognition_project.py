#!/usr/bin/env python
# coding: utf-8

# First check that you have pytorch installed, instructions are here: https://pytorch.org/get-started/locally/ , prefferably do it with anaconda if you can, I think that will lead to less problems down the road if we use other libraries.
# 
# If cuda toolkit isnt available and you have an nvidia gpu try to get that too (it might be contained within anaconda pytorch package): https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
# 
# Also note that because the dataset is large I added it to the .gitignore, you should download it from here : https://www.kaggle.com/c/cassava-leaf-disease-classification/data, and extract it into the data/ folder of the project
# 

# In[1]:


import sys
print(sys.version)


# In[2]:


import torch
import os 
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import torchvision
import pandas as pd
import skorch

from torch import FloatTensor, LongTensor, nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets, models
from IPython.core.debugger import set_trace
#get_ipython().run_line_magic('matplotlib', 'inline')

use_cuda = True
if not torch.cuda.is_available() or not use_cuda:
    print("if you have an nvidia gpu get the cuda core package")
    device = torch.device('cpu')
else:
    print("cuda is available")
    # torch.cuda.set_device(0) # possible fix to illegal memory access error
    device = torch.device('cuda:0')


# Splitting data into train and test sets and loading the validation set

# In[3]:


#setting the path to the directory containing the pics
path = './data/train_images'
test_path = './data/test_images'

labelled_dataset = pd.read_csv(r'./data/train.csv')
submission = pd.read_csv(r'./data/sample_submission.csv')

with open('./data/label_num_to_disease_map.json') as f:
    mapping_dict = json.load(f)
print(mapping_dict)

#labelled_dataset = labelled_dataset.head(250) # tiny dataset for fast debugging, comment when training for real

# Parameters
train, test = train_test_split(labelled_dataset, test_size=0.25, random_state=7, stratify=labelled_dataset.label)

should_match_index = 6
print(labelled_dataset.values[should_match_index])


# In[4]:


from collections import Counter

Counter(labelled_dataset.label) # counts the elements' frequency


# Label Cassava Bacterial Blight (CBB) appears 1087 times<br>
# Label Cassava Brown Streak Disease (CBSD) appears 2189 times<br>
# Label Cassava Green Mottle (CGM) appears 2386 times<br>
# Label Cassava Mosaic Disease (CMD) appears 13158 times<br>
# Label Healthy appears 2577 times<br>
# Because the labels arent equally represented the dataset split is stratified so each split has an equal amount of a certain label
# <br><br>
# Create custom dataset class for the images, and a transform to be applied to these images as part of preprocessing for learning <br>
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# 

# In[5]:


class CassavaDataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# original resolution is 800 x 600
# Parameters
cassava_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600,600)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(299), #minimum is 299 for inceptionv3 224 for everything else
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # (the means and standard deviations of each of the 3 image channels)
])

train_dataset = CassavaDataset(train, path, cassava_transform )
test_dataset = CassavaDataset(test, path, cassava_transform)
valid_dataset = CassavaDataset(submission, test_path, cassava_transform)

print(len(train_dataset))
print(len(test_dataset))
print(len(valid_dataset))
print(len(labelled_dataset))


# In[6]:


train_dataset[0] # how a transformed image tensor looks like, its label is 2


# In[7]:


# Parameters
n_epochs = 15 # on final training this should be high (around 30 for my desktop pc)
num_classes = 5 # 5 labels
batch_size = 28 # minimum batch size for inception v3 is 2, good general range seems to be 20 to 32
learning_rate = 0.0003

# using Adam optimizer, the max batch size for me is around 28, after that it uses too much vram (i have 8gb)
# using SGD optimizer, i can use up to 32
# using different pre processing params, i could get bigger batch sizes since the images would be smaller


train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True, num_workers=0,pin_memory=True,drop_last=True)
valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle=False, num_workers=0,pin_memory=True,drop_last=True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False, num_workers=0,pin_memory=True,drop_last=True)
# pin memory should be enabled if you use cuda cores to speed up transfer between cpu and gpu,
# drop last is there if the last batch contains 1 sample for inception v3 (if its not enabled for inception theres an error)
# num workers is 0 unless you're using linux or mac os x, because paralelization in windows is broken


# Show an image from the dataset

# In[8]:


def show_image(index):  
    plt.imshow(img.imread('{}/{}'.format(path,labelled_dataset.values[index][0]))) # set the correct resolution here
    print('Index: {}'.format(labelled_dataset.values[index][1]))
    print('Filename: {}'.format(labelled_dataset.values[index][0]))
    
show_image(6)


# Using resnet 18 pretrained pytorch model

# In[9]:


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__) 
# Use this number below as the torchvision version for the alternative resnet model or else theres a version conflict


# Model selection below, choices are "resnet", "alexnet", "vgg", "google_net"

# In[10]:


def get_model(model_string):
    if model_string == "google_net":
        net = models.inception_v3(pretrained=True,aux_logits=False) # googlenet is based on inception v1, this is improved
        net = net.cuda() if use_cuda else net
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)

    if model_string == "vgg":
        net = models.vgg16(pretrained=True)
        net = net.cuda() if use_cuda else net
        net.fc = nn.Linear(4096, num_classes)

    if model_string == "alexnet":
        net = models.alexnet(pretrained=True)
        net = net.cuda() if use_cuda else net  
        net.fc = nn.Linear(4096, num_classes)

    if model_string == "resnet":
        #net = torch.hub.load('pytorch/vision:v0.2.2', 'resnet18', pretrained=True) 
        net = models.resnet18(pretrained=True)
        net = net.cuda() if use_cuda else net
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)
        
    net.fc = net.fc.cuda() if use_cuda else net.fc    
    return net
    
def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


# Do evaluation, (still have to try out parameters specified in paper, find out which are useful) <br>The original base code for the block above this text and 2 blocks below was found at the following link, but it was modified for our dataset and to work with our 4 different models not just resnet: https://www.pluralsight.com/guides/introduction-to-resnet <br> <br> What I did with the models is called feature extraction, the models were all pretrained on imagenet: <br>
# 
# In feature extraction, we start with a pretrained model and only update the final layer weights from which we derive predictions. It is called feature extraction because we use the pretrained CNN as a fixed feature-extractor, and only change the output layer.

# In[11]:


# Will plot the accuracy of the models below and save them to a file
def plot_model_acc():
    fig = plt.figure(figsize=(20,10))
    plt.title("Train-Validation Accuracy for {}".format(model_name))
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.savefig('./plots/{}_e{}_lr{}_bs{}.png'.format(model_name,n_epochs,str(learning_rate).replace(".","dot"),batch_size), bbox_inches='tight')


# In[12]:


# This runs all the models, cross validation for all of them is below, this can be used for training and CV for finding optimal parameters
# Parameters
skip_this = True
model_index = 0 # set model index to use here
string_array = ["google_net","alexnet","vgg","resnet"]

first_run = True
while(True and not skip_this):
    if(len(string_array) > 0):
        if not first_run:
            #Print and save graph
            plot_model_acc()

            del criterion # free up vram
            del optimizer
            del net
            torch.cuda.empty_cache()
            model_name = string_array.pop(0)
            net = get_model(model_name)
        else:
            model_name = string_array[model_index]
            net = get_model(model_name)
            string_array.remove(model_name)
            first_run = False

        # Parameters
        criterion = nn.CrossEntropyLoss() # used this since we have 5 mutually exclusive classes
        #either one of the optimizers work
        #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    else:
        plot_model_acc()
        break
    print("-------------------------------------------------------------------")
    print("Using model: {}".format(model_name))
    print_every = int(len(train_dataloader)*0.1) # print upon completion of every 10% of the dataset
    if print_every == 0: print_every = 1
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            #if model_name == "google_net": # for inception v3 (the google model we use): 
                # net(data_) returns logits and aux logits in a touple, we just use logits
                #outputs = outputs[0]
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)

            #set_trace() # Debugger entrypoint for inspecting predictions

            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % print_every == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(test_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')


            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), './data/{}_best_model.pt'.format(model_name))
                print('Improvement-Detected, save-model')
        net.train()


#print(target_,pred) # used for comparing correct class vs models predictions


# Time can vary alot depending on your set parameters, cpu, gpu, whether you're using cuda etc. <br>
# 
# Now instead of rewriting the above boilerplate pytorch code for training and evaluating the model, ill do cross validation using the skorch library, which basically allows you to abstract away the unnecesarry code, do cross validation and hyper parameter tuning with grid search cv

# In[13]:


from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.helper import SliceDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[14]:


import gc
print("Skorch Version: ",skorch.__version__)
#del gs
#del net
gc.collect()
torch.cuda.empty_cache()


# In[15]:


y_train = np.array(train.label)
y_test = np.array(test.label)
params = { # CV testing Parameters, we can add more but keep in mind it takes a very very long time to evaluate all the combinations
    # 'module__dropout': [0, 0.5, 0.8], # 3
    'lr': [0.003,0.00003,0.0000003],# * 3
    'optimizer__weight_decay' :[0.1,0.001,0.00001], # * 3
}
# = 27 total config options * number of folds


# In[ ]:


string_array = ["vgg","google_net","resnet","alexnet"] # also the order


best_params_models = []
# the below is used to cross validate and find the best set of specific parameters to use, it takes a very long time to execute
first_run = True
while(True):
    if(len(string_array)>0):
        if not first_run:
            del skorch_classifier
            del net
            del gs
            del train_sliceable
        model_name = string_array.pop(0)
        net = get_model(model_name)
        gc.collect()
        torch.cuda.empty_cache()
        first_run = False
    else:
        break
    print("---------------------")
    print("---------------------")
    print("Now cross validating {}------------------------------------------".format(model_name))
    print("---------------------")
    print("---------------------")
    skorch_classifier = NeuralNetClassifier(
        net,
        max_epochs=5,
        batch_size=22, #max 22 for me
        lr=0.003, # set in params dict
        #module__dropout=0.5,
        optimizer=torch.optim.SGD,
        optimizer__momentum=0.9,
        optimizer__weight_decay=0.001,
        criterion=nn.CrossEntropyLoss,
        device=device,
        iterator_train__shuffle=True, # Shuffle training data on each epoch
    )
    gs = GridSearchCV(skorch_classifier,
                      param_grid=params, 
                      scoring='accuracy', 
                      verbose=1, # outputs info
                      cv=5, # subsets are still stratified (same percentage of labels)
                     )
    train_sliceable = SliceDataset(train_dataset)
    gs.fit(train_sliceable, y_train)
    best_params_models.append(gs.best_params_)


# In[ ]:


print(best_params_models)
with open('best_params.json', 'w') as jsonfile: # Save best params for each model
    json.dump(best_params_models, jsonfile)


# In[ ]:





# In[ ]:





# In[ ]:




