import torch
import torch.nn
import os
import time
from tqdm import tqdm
from model import DTrOCRLMHeadModel
from config import DTrOCRConfig
from dataset import *
from datetime import datetime
from logger import run

batch_size=4
images,labels,annotations=load_data()
trainset = custom_dataset(images[:-300],labels[:-300],annotations[:-300])
train_loader = DataLoader(trainset, batch_size=batch_size, \
                                    shuffle=True)
valset = custom_dataset(images[-300:],labels[-300:],annotations[-300:])
val_loader = DataLoader(valset, batch_size=batch_size)
config=DTrOCRConfig()
torch.set_float32_matmul_precision('high')
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DTrOCRLMHeadModel(config)
if os.path.exists('/kaggle/input/dtr_ocr/pytorch/default/2/dtrocr_2024-11-06.pth'):
    model.load_state_dict(torch.load('/kaggle/input/dtr_ocr/pytorch/default/2/dtrocr_2024-11-06.pth',weights_only=True,map_location=device))
    print('Loaded weights')
# model = torch.compile(model)
model.to(device)
print(f'Ready for training using {device}.')
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    # set model to evaluation mode
    model.eval()
    
    losses, accuracies = [], []
    with torch.no_grad():
        for inputs in tqdm(dataloader, total=len(dataloader), desc=f'Evaluating test set'):
            inputs = send_inputs_to_device(inputs, device=0)
            outputs = model(**inputs)
            
            losses.append(outputs.loss.item())
            accuracies.append(outputs.accuracy.item())
    
    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    
    # set model back to training mode
    model.train()
    
    return loss, accuracy

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}

use_amp = True
scaler = torch.amp.GradScaler('cuda',enabled=use_amp)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.85, patience=1)

EPOCHS = 2
step_iter=1
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []
for epoch in range(EPOCHS):
    epoch_losses, epoch_accuracies = [], []
    epoch_time = time.time()
    for inputs in tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch + 1}'):
        
        # set gradients to zero
        optimizer.zero_grad()
        
        # send inputs to same device as model
        inputs = send_inputs_to_device(inputs, device)
        
        # forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(**inputs)
        
        # calculate gradients
        scaler.scale(outputs.loss).backward()
        
        # update weights
        scaler.step(optimizer)
        scaler.update()
        
        epoch_losses.append(outputs.loss.item())
        epoch_accuracies.append(outputs.accuracy.item())
        
        #### logger
        run["train/loss"].append(outputs.loss.item())
        run["train/Mean_loss"].append(np.mean(epoch_losses))
        run["train/accuracy"].append(outputs.accuracy.item())
        run["parameters/learning_rate"].append(scheduler.get_last_lr()[0])
        #### logger
        
        if step_iter%200==0:
            torch.save(model.state_dict(),f'dtrocr_{str(datetime.now())[:10]}.pth')
            print('epoch_loss is {:.8f}, epoch_time is {:.8f}, lr is {}'.format(np.mean(epoch_losses), time.time()-epoch_time,scheduler.get_last_lr()[0]))
#             display(gt_geo[0].permute(1,0,2,3))
#             display(pred_geo[0].permute(1,0,2,3))
            scheduler.step(np.mean(epoch_losses))
        
        if step_iter%2000==0:
            validation_loss, validation_accuracy = evaluate_model(model, val_loader)
            run["validation/loss"].append(validation_loss)
            run["validation/accuracy"].append(validation_accuracy)
        step_iter+=1
        
    # store loss and metrics
    train_losses.append(sum(epoch_losses) / len(epoch_losses))
    train_accuracies.append(sum(epoch_accuracies) / len(epoch_accuracies))
torch.save(model.state_dict(),f'dtrocr_{str(datetime.now())[:10]}.pth')