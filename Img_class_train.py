import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def initialize(data_dir):   #data_dir = flowers
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(size = 224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size = 224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size = 224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
}

    image_datasets = {}
    image_datasets["train"] = datasets.ImageFolder(train_dir, transform = data_transforms['train'])
    image_datasets["valid"] = datasets.ImageFolder(valid_dir, transform = data_transforms['valid'])
    image_datasets["test"] = datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    
    train_loaders = torch.utils.data.DataLoader(image_datasets["train"], batch_size = 64, shuffle = True)
    valid_loaders = torch.utils.data.DataLoader(image_datasets["valid"], batch_size = 64, shuffle = True)
    test_loaders = torch.utils.data.DataLoader(image_datasets["test"], batch_size = 64, shuffle = True) 
    
    
    return image_datasets, train_loaders, valid_loaders, test_loaders

def model_arch(arch,hidden_units):
    if arch.lower == "vgg13":
        model = models.vgg13(pretrained=True) 
    else:
        model = models.densenet121(pretrained=True)
        
    for p in model.parameters():
        p.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout1', nn.Dropout(0.1)),
        ('hidden_layer1',nn.Linear(1024,hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout2', nn.Linear(hidden_units,102)),
        ('output',nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    return model 

def train_model(model,train_loaders,valid_loaders,learning_rate,epochs,gpu):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    steps = 0 
    print_every = 10
    accuracy_train = 0
    running_loss = 0
    
    for i in range(epochs):  
        model.train()
        
        for inputs,labels in iter(train_loaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            steps += 1 
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()    
            optimizer.step()
            running_loss += loss.item()
            ps = torch.exp(outputs)
            correct_cnts = (labels.data == ps.max(dim = 1)[1])
            accuracy_train += correct_cnts.type(torch.FloatTensor).mean()
            
            if steps % print_every == 0:
                model.eval()
                validation_accuracy, validation_loss = evaluate_(valid_loaders,criterion,model,gpu)
                print("Epoch {}/{}: ".format(i+1,epochs),
                     "Training Loss: {:.3f}".format(running_loss / print_every),
                      "Training Accuracy: {:.3f}".format((accuracy_train/ print_every)*100),
                     "Validation Accuracy: {:.3f}".format(validation_accuracy),
                      "Validation Loss: {:.3f}".format(validation_loss))
                      
                running_loss = 0
                accuracy_train = 0
                model.train()
                
    print("TRAINING COMPLETE")                
    return model, optimizer, criterion 


def evaluate_(data_loaders,criterion,model,gpu):    
    device = torch.device("cuda:0" if gpu else "cpu")
    correct_cnt = 0
    total = 0
    loss = 0 
    model.to(device)
    with torch.no_grad():
        for inputs, labels in data_loaders:
            
            inputs = inputs.to(device)
            labels = labels.to(device)
             
            outputs = model(inputs)
            ps, predicted = torch.max(outputs.data, 1)
            
            loss = criterion(outputs,labels).item()
            
            total += labels.size(0)
            correct_cnt += (predicted == labels).sum().item()

            accuracy = (100 * correct_cnt / total)
            return accuracy, loss
        
parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', action = 'store', default = '.', dest = 'save_dir', help = 'Directory to save checkpoints')
parser.add_argument('--arch', action = 'store', default = 'densenet121', dest = 'arch', help = 'Architecture to be used eg "vgg16"')
parser.add_argument('--learning_rate', action = 'store', default = 0.001, dest = 'learning_rate', help = 'Architectures learning rate')
parser.add_argument('--hidden_units', action = 'store', default = 512, dest = 'hidden_units', help = 'Choose hidden units')
parser.add_argument('--epochs', action = 'store', default = 15, dest = 'epochs', help = 'Choose the number of epochs')
parser.add_argument('--gpu', action = 'store_true', default = 'False', dest = 'gpu', help = 'To use gpu for training, switch to True')

args = parser.parse_args()

save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

data_dir = 'flowers'

image_datasets, train_loader, valid_loader, test_loader = initialize(data_dir)

model = model_arch(arch,hidden_units)

# def train_model(model,train_loaders,valid_loaders,learning_rate,epochs,device):
model_trained, optimizer, criterion = train_model(model,train_loader,valid_loader,lr,epochs,gpu)

#saving checkpoint
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'input_size' : 224*224*3,
    'outputs' : 102,
    'model' : model,
    'state_dict':model.state_dict(),
    'optimnizer_state_dict': optimizer.state_dict,
    'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')