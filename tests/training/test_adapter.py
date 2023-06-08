import os
from transformers import AutoTokenizer, XLMRobertaModel
import torch
from transformers.adapters import XLMRobertaAdapterModel, AutoAdapterModel, BertAdapterModel, AdapterConfig

#from torchinfo import summary

print('imported')
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")


# model = AutoAdapterModel.from_pretrained("xlm-roberta-base")
config = AdapterConfig.load("pfeiffer")

print('model loaded')

sentences = ["Hello, my dog is cute", "Hello, my dog is cute", "Hello, my dog is cute"]
# input = torch.as_tensor(sentences)

print(type(input))

inputs = tokenizer(sentences, return_tensors="pt")
print(inputs.input_ids.size())
print(f'inputs : {inputs}')
outputs = model(**inputs, return_tensors="pt")
print(f'outputs : {outputs}')
print(f"output type is {type(outputs)}")
# last_hidden_states = outputs.last_hidden_state
# print(f"last hidden states : {last_hidden_states}")

# example_path = os.path.join(os.getcwd(), "adapter-quickstart")
# adapter_name = 'test adapter'
got_adapters = model.has_adapters()
print(f"Presence of adapters : {got_adapters}")

# model.load_adapter("argument/ukpsent@ukp", config=config)
# model.load_adapter('my_adapter', config=config)
model.add_adapter("my_adapter", config=config)

got_adapters = model.has_adapters()
print(f"Presence of adapters : {got_adapters}")

# # Save model
# model.save_pretrained(example_path)
# # Save adapter
# model.save_adapter(example_path, adapter_name)

# # Load model, similar to HuggingFace's AutoModel class, 
# # you can also use AutoAdapterModel instead of BertAdapterModel
# model = AutoAdapterModel.from_pretrained(example_path)
# model.load_adapter(example_path)

# model_hg = XLMRobertaModel.from_pretrained("xlm-roberta-base")
# model_ah = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")


# summary(model_hg, (1, 1, 8))
# print("-------------------------------------")
# summary(model_ah, (1, 1, 8))


## ---------------------------------------------------------------------------------

# import torch 

# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim

# from torch.optim.lr_scheduler import ExponentialLR




# try :
#     # Device configuration
#     #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = 'cpu'

#     ## We set a seed manually so as to reproduce the results easily
#     seed  = 2147483647
#     torch.manual_seed(seed)

#     # ------------------------------------------



#     ## Hyperparameters
#     num_classes = 10
#     num_epochs = 120
#     batch_size = 64
#     learning_rate = 0.001
#     weight_decay = 10e-4
#     momentum = 0.9

#     ## Defining training and dataset
#     training = 'mixup'
#     dataset = 'cifar10'

#     ## Base directory from EFDL to EFDL_storage
#     base_dir = '../EFDL_storage'

#     # The data will be downloaded in the following folder
#     rootdir = base_dir+'/data/'+dataset

#     # adapt the set for test
#     train_dataset = ''
#     dev_dataset = ''

#     trainloader = DataLoader(c10train,batch_size=batch_size,shuffle=True)
#     testloader = DataLoader(c10test,batch_size=batch_size) 

#     ## number of target samples for the final dataset
#     num_train_examples = len(c10train)
#     num_samples_subset = 15000

#     # Model definition
#     tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
#     model_hg = XLMRobertaModel.from_pretrained("xlm-roberta-base")
#     model_ah = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
   
#     print('Beginning of training')

#     # Loss and optimizer
#     end_sched = max(int(4*num_epochs/5), 100)
#     criterion = nn.CrossEntropyLoss()
#     optimizer_hg = optim.AdamW( model_hg.parameters(),
#                             lr=learning_rate,
#                             weight_decay = weight_decay, # if no weight decay it means we are regularizing
#                             momentum = momentum) 
#     optimizer_ah = optim.AdamW( model_ah.parameters(),
#                         lr=learning_rate,
#                         weight_decay = weight_decay, # if no weight decay it means we are regularizing
#                         momentum = momentum) 
#     scheduler = CosineAnnealingLR(optimizer,
#                                 T_max = end_sched, # Maximum number of iterations.
#                                 eta_min = learning_rate/100) # Minimum learning rate. 
   


#     # To plot the accruacy
#     epoch_list = list(range(num_epochs))

#     running_loss = 0.
#     train_losses = []
#     val_losses = []

#     correct = 0
#     total = 0
#     train_acc = []
#     val_acc = []

#     # Train the model
#     total_step_train = len(trainloader)
#     total_step_val = len(testloader)
#     number_batch = len(trainloader)

#     for epoch in range(num_epochs):
        
#         print(f'Epoch [{epoch+1}/{num_epochs}]')

#         running_loss = 0.
#         correct = 0
#         total = 0

#         model.train()
#         for i, (images, labels) in enumerate(trainloader) :  
#             # Move tensors to the configured device
#             images = images.to(device=device, dtype=torch.float32)
#             labels = labels.to(device=device, dtype=torch.long)

#             # Forward pass
#             outputs = model(images)

#             # Compute loss
#             loss = criterion(outputs, labels)
#             running_loss += loss.item() # pour calculer sur une moyenne d'epoch
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # For accuracy (up : classic, down : mixup)
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == labels).float().sum().item()
#             total += labels.size(0)

#             print("\r"+"Batch training : ",i+1,"/",number_batch ,end="")

#             torch.cuda.empty_cache()
        
#         # del images, labels, outputs
#         # torch.cuda.empty_cache()

#         train_losses.append(running_loss / total)
#         train_acc.append(100*correct/total)
#         print('\n'+f'Train Loss : {train_losses[-1]:.4f} , Train accuracy : {train_acc[-1]:.4f}')


#         with torch.no_grad():

#             running_loss = 0.
#             correct = 0
#             total = 0

#             model.eval()
#             for i, (images, labels) in enumerate(testloader) :
#                 # Move tensors to the configured device
#                 images = images.to(device)
#                 labels = labels.to(device)

#                 # Forward pass
#                 outputs = model(images)

#                 # Compute loss
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item() # pour calculer sur une moyenne d'epoch

#                 # For accuracy
#                 _, predicted = torch.max(outputs.data, 1)
#                 correct += (predicted == labels).float().sum().item()
#                 total += labels.size(0)

#             val_losses.append(running_loss / total)
#             val_acc.append(100*correct/total)

#             # del images, labels, outputs
#             # torch.cuda.empty_cache()

#         print(f'Validation loss: {val_losses[-1]:.4f} , Validation accuracy: {val_acc[-1]:.4f}')

#         # Early stopping in case of overfitting
#         if early_stopper.early_stop(running_loss):

#             model_dir_early = base_dir+'/models/'+ model_name +'_'+ training +'_'+ dataset +'_epoch'+str(epoch)+'_early.pt'
#             model_state = {'model name': model_name,
#                         'model': model,
#                         'optimizer': optimizer,
#                         'epoch': epoch,
#                         'training': training,
#                         'dataset': dataset,
#                         'metrics' : {'train_loss' : train_losses,
#                                      'val_loss' :val_losses,
#                                      'train_acc' :train_acc,
#                                      'val_acc' : val_acc}
#                         }
#             torch.save(model_state, model_dir_early)
#             print("\n"+"Training stop early at epoch ",epoch+1,"/",num_epochs," with a loss of : ",running_loss/total,", and accuracy of : ",100*correct/total)
#             stopping_list.append(epoch+1)
        
#         print('---------------------------------------')

#     if len(stopping_list) == 0:
#         print("Pas d'overfiting !")
#     else :
#         print("\n"+"Les epoch d'early stop sont : ",stopping_list)

#     # ------------------------------------------
#     # save the model and weights
#     model_state = {'model name': model_name,
#             'model': model,
#             'optimizer': optimizer,
#             'epoch': num_epochs,
#             'training': training,
#             'dataset': dataset,
#             'metrics' : {'train_loss' : train_losses,
#                                      'val_loss' :val_losses,
#                                      'train_acc' :train_acc,
#                                      'val_acc' : val_acc}
#             }
#     torch.save(model_state, model_dir)
#     print("Modèle sauvegardé dans le chemin : ",model_dir)


# except KeyboardInterrupt:
#     print("\nKeyboard interrupt, we have saved the model and its metrics")

#     model_state = {'model name': model_name,
#         'model': model,
#         'optimizer': optimizer,
#         'epoch': num_epochs,
#         'training': training,
#         'dataset': dataset,
#         'metrics' : {'train_loss' : train_losses,
#                                      'val_loss' :val_losses,
#                                      'train_acc' :train_acc,
#                                      'val_acc' : val_acc}
#         }
#     torch.save(model_state, model_dir)
#     print("Modèle sauvegardé dans le chemin : ",model_dir)

# finally:
#     print("end of training script of ",model_name)