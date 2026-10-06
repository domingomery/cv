import torch
import matplotlib.pyplot as plt
from   torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data.dataloader import DataLoader
from   sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
import time



def display_img(train_set,k):
    (img,label) = train_set[k]
    print(f"Image: {k}, Label : {train_set.classes[label]}")
    plt.imshow(img.permute(1,2,0))

def show_batch(dl,nrow=16):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=nrow).permute(1,2,0))
        break

class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def prediction_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        return out

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result, st):
        x = np.zeros((3,))
        x[0] = result['train_loss']
        x[1] = result['val_loss']
        x[2] = result['val_acc']
        print("%5d %11.4f %11.4f %11.4f %s" % (epoch, x[0], x[1], x[2],st))



def prediction(model, val_loader):
    model.eval()
    pred = [model.prediction_step(batch) for batch in val_loader]
    return pred

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD, earlystop=10):
    acc_max = 0
    history = []
    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()    
    for epoch in range(epochs):
        t1 = time.time()    
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        acc = result['val_acc']
        st = '          '
        if acc>acc_max:
          epoch_max = epoch
          acc_max = acc
          st = '   ***    '
          torch.save(model.state_dict(),'best_model.pt')
        t2 =  "{:6.1f} ".format(time.time() - t1)
        model.epoch_end(epoch, result,st+t2)
        history.append(result)
        if epoch - epoch_max >= earlystop:
            print(rs)
            print('*** Early stop after '+str(earlystop)+' epochs with no improvement')
            break

    print(rs)
    torch.save(model.state_dict(),'last_model.pt')
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    return history

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('val-accuracy')
    plt.title('Accuracy vs. No. of epochs');
    
def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

def get_prediction(model,dataset):
  set_dl = DataLoader(dataset, len(dataset), shuffle = False, num_workers = 2, pin_memory = True)
  y      = prediction(model,set_dl)
  y1     = torch.cat(y)
  y2     = y1.detach().numpy()
  ypred  = np.argmax(y2,axis = 1)
  return ypred        

def get_labels(model,dataset):
  set_dl = DataLoader(dataset, len(dataset), shuffle = False, num_workers = 2, pin_memory = True)
  for batch in set_dl:
    img,label = batch
  ygt = label.numpy()
  return ygt  

def performance(model,dataset,st,show=1):
  ypred = get_prediction(model,dataset)  
  ygt   = get_labels(model,dataset)
  C     = confusion_matrix(ygt,ypred)
  acc   = accuracy_score(ygt,ypred) 
  if show == 1:
    print(st+' Confusion Matrix = ')
    print(C)
    print(' ')
    print("{} Accuracy = {:.4f}".format(st,acc))
    print(' ')

  return C,acc        

def load_model(CNN_model,model_file):
    model =  CNN_model()
    model.load_state_dict(torch.load(model_file))
    print(model_file+' loaded.')
    return model

def print_confusion(dt,ds,show_heatmap=0,Cnorm=1):
    # dt: GT, ds: Prediction
    C   = confusion_matrix(dt,ds) 
    print('Confusion Matrix:')
    print(C)
    acc = accuracy_score(dt,ds) 
    acc_st = "{:.2f}".format(acc*100)
    print('Accuracy = '+str(acc_st))
    if show_heatmap:
      sns.heatmap(C/Cnorm, annot=True, cbar=None, cmap="Blues")
      plt.title("Confusion Matrix: Acc ="+acc_st)
      plt.tight_layout()
      plt.ylabel("True Class"), plt.xlabel("Predicted Class")
      plt.show()
