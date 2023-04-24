import torch
import sys
import torch.nn as nn
import torch.optim as optim

# Rq : Adapt the function for the CEA internship

def keybInterrupt():
# def keybInterrupt(model: nn.Module, model_name: str, optimizer: optim,
#                   num_epochs: int, training: str, dataset: str,
#                   model_dir: str, 
#                   val_losses: list, val_acc: list, 
#                   train_losses: list, train_acc: list=None)-> None:
    '''Prompt if nedds to ave the model and accuracy through time'''
    
    prompt = input("\nKeyboard interrupt, do you want to the model and its metrics ? \nyes/no \n")
    if (prompt=='yes' or prompt=='y' or prompt=='Y'):
        if train_acc == None :
            model_state = {'model name': model_name,
                'model': model,
                'optimizer': optimizer,
                'epoch': num_epochs,
                'training': training,
                'dataset': dataset,
                'metrics' : {'train_loss' : train_losses,
                            'val_loss' :val_losses,
                            'val_acc' : val_acc}
                }
        else :
            model_state = {'model name': model_name,
                'model': model,
                'optimizer': optimizer,
                'epoch': num_epochs,
                'training': training,
                'dataset': dataset,
                'metrics' : {'train_loss' : train_losses,
                            'train_acc' : train_acc,
                            'val_loss' :val_losses,
                            'val_acc' : val_acc}
                }
        torch.save(model_state, model_dir)
        print("Model saved in the dir : ",model_dir)
    else :
        print("Model not saved")
    return None