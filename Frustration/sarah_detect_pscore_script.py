''' script to predict PSPI from image '''
from __future__ import print_function
from __future__ import division
import sys
sys.path.insert(0, './../../../')
sys.path.insert(0, './../../')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import RacialImageDataset
import imp
import sklearn.metrics
imp.reload(RacialImageDataset)
from RacialImageDataset import *
from vgg_face import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def set_rseed(rseed):
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    np.random.seed(rseed)
    random.seed(rseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "NN1":
        """ 1-hidden layer NN
        """
        input_size = 9
        model_ft = NN1(input_size, 2 * input_size, num_classes)

    elif model_name == "face1_vgg":
        """ vgg-vd-16 trained on vggface
        """
        model_ft = VGG_16()
        if use_pretrained:
            model_ft.load_weights()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc8.in_features
        model_ft.fc8 = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def test_model(model, dataloader):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_pred_label = np.empty((0,11))

    # Iterate over data.
    for sample in dataloader:
        inputs = sample['image'] # 4D tensor ---> 8 x 3 x 224 x 224
        labels = sample['label'].reshape(-1,1).float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        imagedirs = sample['image_dir'] # list of string
        imageids = sample['image_id'] # list of string
        print(imageids)

        # forward
        # track history if only in train
        last_layer = list(model_ft.children())[-1]
        last2_layer = list(model_ft.children())[-3] #because of dropout [-2]
        try:
            last_layer = last_layer[-1]
            last2_layer = last_layer[-3]
        except:
            last_layer = last_layer
            last2_layer = last2_layer
        my_embedding = torch.zeros((inputs.shape[0],last_layer.in_features))
        my_embedding2 = torch.zeros((inputs.shape[0],last2_layer.in_features))
        def fun(m, i, o):
            my_embedding.copy_(i[0].data)
        def fun2(m, i, o):
            my_embedding2.copy_(i[0].data)
        h = last_layer.register_forward_hook(fun)
        h2 = last2_layer.register_forward_hook(fun2)
        with torch.set_grad_enabled(False):
            # Get model outputs
            outputs = model(inputs) * torch.FloatTensor([16] + [5]*9).to(device)
            print(outputs[0])
        h.remove()
        h2.remove()
        # save results
        for curr_id in range(len(imageids)):
            currdir=imagedirs[curr_id]
            label = np.zeros((10))
            """
            splt = imageids[curr_id].split('_')
            for spl in splt:
              sp = spl.split('.')
              if sp[0][0]=='a':
                au=sp[0][2:]
                pspi =0
                if au == '4':
                    label[1] = int(sp[1])/10
                    pspi+= (int(sp[1])/2)
                elif au == '6':
                    label[2] = int(sp[1])/10
                elif au == '7':
                    label[3] = int(sp[1])/10
                elif au == '10':
                    label[4] = int(sp[1])/10
                    pspi+= (int(sp[1])/2)
                elif au=='12':
                    label[5] = int(sp[1])/10
                elif au=='20':
                    label[6] = int(sp[1])/10
                elif au=='25':
                    label[7] = int(sp[1])/10
                elif au=='26':
                    label[8] = int(sp[1])/10
                elif au=='43':
                    label[9] = int(sp[1])/10
                    pspi+= (int(sp[1])/10)
                if label[2] > label[3]:
                    pspi+= (label[2]*5)
                else:
                    pspi+= (label[3]*5)
            pspi = pspi/16
            label[0] = pspi
            """
            #f=open(os.path.join(currdir, imageids[curr_id][:-4] + '.txt'), "w")
            #f.write(str(outputs.data.cpu().numpy()[curr_id][0]))
            #f.close()

        #exit()
        # statistics
        #running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))
    #pred_test = running_pred_label[:,0:1]
    #label_test = running_pred_label[:,10:]

    #sampleweights = 1
    #epoch_acc = (np.round(pred_test) == label_test).mean(0)[0]
    #epoch_weighted_acc = ((np.round(pred_test) == label_test) * sampleweights).mean(0)[0]
    #epoch_mse = ((pred_test - label_test)**2).mean(0)[0]
    #epoch_weighted_mse = ((pred_test - label_test)**2 * sampleweights).mean(0)[0]
    #epoch_mae = np.abs(pred_test - label_test).mean(0)[0]
    #epoch_weighted_mae = (np.abs(pred_test - label_test) * sampleweights).mean(0)[0]

    #print('{} Acc: {:.4f} Weighted Acc: {:.4f} MSE: {:.4f} Weighted MSE: {:.4f} MAE: {:.4f} Weighted MAE: {:.4f}'.format('test',
    #        epoch_acc, epoch_weighted_acc, epoch_mse, epoch_weighted_mse, epoch_mae, epoch_weighted_mae))

    #f=open('BestEpoch.txt', "a")
    #f.write('Test MAE: {:4f} weighted MAE {:4f} \n'.format(epoch_mae, epoch_weighted_mae))
    #f.close()

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #return pred_test, label_test

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import random

image_dirs = [
    '/home/AD/rraina/pain/Frustration/generated_faces/AU25_experiments/morpha_2shoulder_zscore_hsv_diffexpFalse_excludewhiteTrue_background255/'
]
for image_dir in image_dirs:
        rseed = 0
        set_rseed(rseed)

        # image_dir = './../../../data/racial_pain/Stimuli/Experiment5/BlackTargets/'
        model_dir = '/mnt/cube/projects/xiaojing/shoulder_pain_detection_weightall/newnorm_PSPIAU/models_sf1/'
        #model_dir = './../../../shoulder_pain_detection_weightall/newnorm_PSPIAUpersubj/newnorm_PSPIAU/models_sf0/2115.pth'
        # Number of classes in the dataset
        num_classes = 10
        # Batch size for training (change depending on how much memory you have)
        batch_size = 1
        # Number of epochs to train for
        num_epochs = 100
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = False
        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        model_name = "face1_vgg"

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Print the model we just instantiated
        print(model_ft)

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transform = transforms.Compose([
                # BGR2RGB(),
                darken(0.5),
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize([0.3106, 0.3515, 0.5092], [0.16951026,  0.19180746, 0.2778062]),
                transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
            ])

        print("Initializing Datasets and Dataloaders...")

        # start testing
        for fold in range(1):#5):
            model_name = os.path.join(model_dir, f"{fold}.pth") #not needed for individual models
            #model_name = model_dir
            dataset = RacialImageDataset(image_dir, data_transform)
            dataloaders_dict = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=lambda l: [np.random.seed((rseed + l)), random.seed(rseed + l), torch.manual_seed(rseed+ l)])

            # Train and evaluate
            model_ft.load_state_dict(torch.load(model_name,map_location=device))

            # test
            test_model(model_ft, dataloaders_dict)