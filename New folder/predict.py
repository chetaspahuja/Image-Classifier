import argparse
import json
import PIL
import torch 
import numpy as np
from torchvision import models
from math import ceil
from PIL import Image

def process_image(image_path):
    #Resizing 
    image = Image.open(image_path)
    w,h = image.size
    
    if w > h:
        image = image.resize((round((w/h)*256),256))
    else:
        image = image.resize((256,round(256/(w/h))))
                       
    #Cropping 
    w1,h1 = image.size
    new_w = 224
    new_h = 224 
    
    left = (w1-new_w)/2
    right = (w1+new_w)/2
    top = (h1-new_h)/2
    bottom = (h1+new_h)/2
                             
    image = image.crop((round(left),round(top),round(right),round(bottom)))
                             
    #normalizing, reordering dimensions 
    
    np_image = np.array(image) /225
    np_image = (np_image - np.array([0.485,0.456,0.406])/np.array([0.229,0.224,0.225]))
    np_image = np_image.transpose((2,0,1))
    
    return np_image # -> returning value = tensor 

def predict(np_image, device, model, topk = 5):
#     device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        imgs = torch.from_numpy(np_image)
        imgs = imgs.unsqueeze(0)
        imgs = imgs.type(torch.FloatTensor)
        imgs = imgs.to(device)
        out = model.forward(imgs)
        ps = torch.exp(out)
        pbs, inds = torch.topk(ps,topk)
        pbs = [float(pb) for pb in pbs[0]]
        inv_map = {val:key for key, val in model.class_to_idx.items()}
        clss = [inv_map[int(idx)] for idx in inds[0]]
        return pbs, clss    
    

def model_load(file_path):
    checkpoint = torch.load(file_path, map_location = 'cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model


parser = argparse.ArgumentParser()

parser.add_argument('image_path', action = 'store', help = 'Add path to image to be predicted')
parser.add_argument('checkpoint', action = 'store', default='.', help = 'Directory of saved checkpoints')
parser.add_argument('--topk', action = 'store', type = int,default= 5, help = 'Returns top K classes ')
parser.add_argument('--category_names', action='store', default = 'cat_to_name.json', dest='category_names', help='File name of the mapping of flower categories to real names')
parser.add_argument('--gpu', action='store', default= 'CPU', dest='gpu', help='Use GPU for inference, enter "CPU" or "GPU" ')

args = parser.parse_args()
image_path = args.image_path
checkpoint = args.checkpoint
topk = args.topk 
category_names = args.category_names
gpu = args.gpu

if gpu == "CPU":
    device = 'cpu'
else:
    device = 'cuda' 

#mapping labels
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
file_path = checkpoint 
model = model_load(file_path)

np_image = process_image(image_path)

probabilties, classes = predict(np_image,device,model,topk)

name_classes = []

for i in classes:
    name_classes.append(cat_to_name[i]) 
    
print("Flower Probability -> ")

for i in range(len(probabilties)):
    print(name_classes[i], ": " , round(probabilties[i]*100,5)) 
    
print("")





    



