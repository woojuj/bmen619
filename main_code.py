
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import torch.optim.lr_scheduler as lr_scheduler
import os
from tqdm.auto import tqdm

# job_id for comparing other results
import sys
if len(sys.argv) > 1:
    job_id = sys.argv[1]
else:
    job_id = 0

# Define dataset paths (512 resolution)
image_path_train = '/home/wooju.chung/BCSS_dataset/BCSS_512/train_512/'
mask_path_train = '/home/wooju.chung/BCSS_dataset/BCSS_512/train_mask_512/'

image_path_val = '/home/wooju.chung/BCSS_dataset/BCSS_512/val_512/'
mask_path_val = '/home/wooju.chung/BCSS_dataset/BCSS_512/val_mask_512/'

image_path_test = '/home/wooju.chung/BCSS_dataset/BCSS_512/test_512/'
mask_path_test = '/home/wooju.chung/BCSS_dataset/BCSS_512/test_mask_512/'

# Define dataset pathss (224 resolution)
# image_path_train = '/home/wooju.chung/BCSS_dataset/BCSS/train/'
# mask_path_train = '/home/wooju.chung/BCSS_dataset/BCSS/train_mask/'

# image_path_val = '/home/wooju.chung/BCSS_dataset/BCSS/val/'
# mask_path_val = '/home/wooju.chung/BCSS_dataset/BCSS/val_mask/'

# image_path_test = '/home/wooju.chung/BCSS_dataset/BCSS/test/'
# mask_path_test = '/home/wooju.chung/BCSS_dataset/BCSS/test_mask/'

# Set number of output channels based on dataset resolution
models_number= 22
#22 is 512, 3 is 224

# Define a function to create a list of the paths of the images and masks.
def image_mask_path(image_path:str, mask_path:str):
    IMAGE_PATH = Path(image_path)
    IMAGE_PATH_LIST = sorted(list(IMAGE_PATH.glob("*.png")))

    MASK_PATH = Path(mask_path)
    MASK_PATH_LIST = sorted(list(MASK_PATH.glob("*.png")))
    
    return IMAGE_PATH_LIST, MASK_PATH_LIST


IMAGE_PATH_LIST_TRAIN, MASK_PATH_LIST_TRAIN = image_mask_path(image_path_train, 
                                                              mask_path_train)
IMAGE_PATH_LIST_VAL, MASK_PATH_LIST_VAL = image_mask_path(image_path_val, 
                                                          mask_path_val)

print(f'Total Images Train: {len(IMAGE_PATH_LIST_TRAIN)}')
print(f'Total Masks Train: {len(MASK_PATH_LIST_TRAIN)}')
print(f'Total Images Val: {len(IMAGE_PATH_LIST_VAL)}')
print(f'Total Masks Val: {len(MASK_PATH_LIST_VAL)}')

VALUES_UNIQUE_TRAIN = []

for i in MASK_PATH_LIST_TRAIN:
    sample = cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
    uniques = np.unique(sample)
    VALUES_UNIQUE_TRAIN.append(uniques)
    
FINAL_VALUES_UNIQUE_TRAIN = np.concatenate(VALUES_UNIQUE_TRAIN)
print("Unique values Train:")
print(np.unique(FINAL_VALUES_UNIQUE_TRAIN))

# 
VALUES_UNIQUE_VAL = []

for i in MASK_PATH_LIST_VAL:
    sample = cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
    uniques = np.unique(sample)
    VALUES_UNIQUE_VAL.append(uniques)
    
FINAL_VALUES_UNIQUE_VAL = np.concatenate(VALUES_UNIQUE_VAL)
print("Unique values Validation:")
print(np.unique(FINAL_VALUES_UNIQUE_VAL))


# Display images and masks separately
fig, ax = plt.subplots(nrows = 10, ncols = 2, figsize = (20,30))

for i,(img_path, mask_path) in enumerate(zip(IMAGE_PATH_LIST_TRAIN, MASK_PATH_LIST_TRAIN)):
    if i>9:
        break
        
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax[i,0].imshow(img_rgb)
    ax[i,0].axis('off')
    ax[i,0].set_title(f"Image\nShape: {img_rgb.shape}", fontsize = 10, fontweight = "bold", color = "black")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ax[i,1].imshow(mask)
    ax[i,1].axis('off')
    ax[i,1].set_title(f"Mask\nShape: {mask.shape}", fontsize = 10, fontweight = "bold", color = "black")

fig.tight_layout()
# fig.savefig(f"/home/wooju.chung/bmen619/figure/other_sample_{job_id}.png")
plt.close(fig)

# Visualize images with masks overlaid
fig, ax = plt.subplots(nrows = 10, ncols = 2, figsize = (12,30))
ax = ax.flat

for i,(img_path, mask_path) in enumerate(zip(IMAGE_PATH_LIST_TRAIN, MASK_PATH_LIST_TRAIN)):
    
    if i>19:
        break
        
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax[i].imshow(img_rgb)
    ax[i].axis('off')
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ax[i].imshow(mask, alpha = 0.30)
    ax[i].axis('off')
    

fig.tight_layout()
# fig.savefig(f"/home/wooju.chung/bmen619/figure/other_mask_superimposed_{job_id}.png")
plt.close(fig)

# Create dataframes for training and validation datasets
data_train = pd.DataFrame({'Image':IMAGE_PATH_LIST_TRAIN, 
                           'Mask':MASK_PATH_LIST_TRAIN})

data_val = pd.DataFrame({'Image':IMAGE_PATH_LIST_VAL, 
                         'Mask':MASK_PATH_LIST_VAL})

# Preprocessing function based on pretrained ResNet34
preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name = "resnet34", 
                                        pretrained = "imagenet")

# Define transformations for training, validation, and test datasets
transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(30),     
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

transforms_val_test = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_train_mask = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(30),     
    transforms.RandomVerticalFlip(p=0.5),   
    transforms.PILToTensor()           
])

transforms_val_mask = transforms.Compose([
    transforms.PILToTensor(),  
])

# Dataset Definition 
class CustomImageMaskDataset(Dataset):
    def __init__(self, data:pd.DataFrame, image_transforms, mask_transforms):
        self.data = data
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)
        
        mask_path = self.data.iloc[idx, 1]
        mask = Image.open(mask_path)
        mask = self.mask_transforms(mask)
        
        return image, mask

#Dataset Preparation
train_dataset = CustomImageMaskDataset(data_train, transforms_train, 
                                    transforms_train_mask)
    
val_dataset = CustomImageMaskDataset(data_val, transforms_val_test, 
                                    transforms_val_mask)


# DataLoader

BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()
print(NUM_WORKERS, ': NUM_WORKERS')

train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, 
                            shuffle = True, num_workers = NUM_WORKERS)

val_dataloader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, 
                            shuffle = False 
                            , num_workers = NUM_WORKERS)


batch_images, batch_masks = next(iter(train_dataloader))
batch_images.shape, batch_masks.shape

batch_images, batch_masks = next(iter(train_dataloader))
batch_images.shape, batch_masks.shape

# CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# Initializing U-Net model
model = smp.Unet(in_channels = 3, classes = models_number)


for param in model.encoder.parameters():
    param.requires_grad = False

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005 # 0.001
                       , weight_decay = 0.0001) #l2 


scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True) 


# Training
# EarlyStopping: Stops training when validation loss does not improve.
# train_step: Performs a single training step.
# val_step: Performs a single validation step.
# predictions_mask: Generates mask predictions.

class EarlyStopping:
    def __init__(self, patience:int = 5, delta:float = 0.0001, path = f"/home/wooju.chung/bmen619/model/best_model_{job_id}.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
            
        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# Define early stopping
early_stopping = EarlyStopping(patience = 10, delta = 0.)

def train_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer):
    
    model.train()
    
    train_loss = 0.
    train_accuracy = 0.
    
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device = DEVICE, dtype = torch.float32)
        y = y.to(device = DEVICE, dtype = torch.long)
        optimizer.zero_grad()
        logit_mask = model(X)
        loss = loss_fn(logit_mask, y.squeeze())
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        prob_mask = logit_mask.softmax(dim = 1)
        pred_mask = prob_mask.argmax(dim = 1)
        
        tp,fp,fn,tn = smp.metrics.get_stats(output = pred_mask.detach().cpu().long(), 
                                            target = y.squeeze().cpu().long(), 
                                            mode = "multiclass", 
                                            num_classes = models_number)
        
        train_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction = "micro").numpy()
        
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy

def val_step(model:torch.nn.Module, 
             dataloader:torch.utils.data.DataLoader, 
             loss_fn:torch.nn.Module):
    
    model.eval()
    
    val_loss = 0.
    val_accuracy = 0.
    
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X = X.to(device = DEVICE, dtype = torch.float32)
            y = y.to(device = DEVICE, dtype = torch.long)
            logit_mask = model(X)
            loss = loss_fn(logit_mask, y.squeeze())
            val_loss += loss.item()
            
            prob_mask = logit_mask.softmax(dim = 1)
            pred_mask = prob_mask.argmax(dim = 1)
            
            tp, fp, fn, tn = smp.metrics.get_stats(output = pred_mask.detach().cpu().long(), 
                                                   target = y.squeeze().cpu().long(), 
                                                   mode = "multiclass", 
                                                   num_classes = models_number) #224 여기도 변경? 
            
            val_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction = "micro").numpy()
            
    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader)
    
    return val_loss, val_accuracy

# Training

def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, 
          val_dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, 
          optimizer:torch.optim.Optimizer, scheduler, early_stopping, epochs:int = 10): # lr 
    
    results = {'train_loss':[], 'train_accuracy':[], 'val_loss':[], 'val_accuracy':[]}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model = model, 
                                           dataloader = train_dataloader, 
                                           loss_fn = loss_fn, 
                                           optimizer = optimizer)
        
        val_loss, val_accuracy = val_step(model = model, 
                                     dataloader = val_dataloader, 
                                     loss_fn = loss_fn)
        
        print(f'Epoch: {epoch + 1} | ', 
              f'Train Loss: {train_loss:.4f} | ', 
              f'Train Accuracy: {train_accuracy:.4f} | ', 
              f'Val Loss: {val_loss:.4f} | ', 
              f'Val Accuracy: {val_accuracy:.4f}')
        
        early_stopping(val_loss, model)
        scheduler.step(val_loss)
        
        if early_stopping.early_stop == True:
            print(f"Early Stopping at epoch {epoch + 1}, patience was {early_stopping.patience}!!!")
            break
            
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_accuracy)
        
    return results


# Training Execution

SEED = 42
EPOCHS = 40
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

RESULTS = train(model.to(device = DEVICE), 
                train_dataloader, 
                val_dataloader, 
                loss_fn, 
                optimizer, 
                scheduler, # lr
                early_stopping, 
                EPOCHS)

# Metrics Visualization
def loss_and_metric_plot(results:dict):
    
    training_loss = results['train_loss']
    training_metric = results['train_accuracy']
    
    validation_loss = results['val_loss']
    validation_metric = results['val_accuracy']
    
    fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9,3.8))
    ax = ax.flat
    
    ax[0].plot(training_loss, label = "Train")
    ax[0].plot(validation_loss, label = "Val")
    ax[0].set_title("CrossEntropyLoss", fontsize = 12, fontweight = "bold", color = "black")
    ax[0].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[0].set_ylabel("loss", fontsize = 10, fontweight = "bold", color = "black")
    ax[0].legend()  # 범례 추가
    
    ax[1].plot(training_metric, label = "Train")
    ax[1].plot(validation_metric, label = "Val")
    ax[1].set_title("Accuracy", fontsize = 12, fontweight = "bold", color = "black")
    ax[1].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[1].set_ylabel("score", fontsize = 10, fontweight = "bold", color = "black")
    ax[1].legend()  # 범례 추가


    fig.tight_layout()
    fig.savefig(f"/home/wooju.chung/bmen619/figure/other_loss_acc_{job_id}.png")
    plt.close(fig)

loss_and_metric_plot(RESULTS)

# Final Predictions

def predictions_mask(test_dataloader:torch.utils.data.DataLoader):
    # Load the best model checkpoint
    checkpoint = torch.load(f"/home/wooju.chung/bmen619/model/best_model_{job_id}.pth")
    loaded_model = smp.Unet(encoder_weights = None, classes = models_number)
    loaded_model.load_state_dict(checkpoint)
    loaded_model.to(device = DEVICE)
    loaded_model.eval()

    y_pred_mask = []

    with torch.inference_mode():
        for batch,X in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
            X = X.to(device = DEVICE, dtype = torch.float32)
            mask_logit = loaded_model(X)
            mask_prob = mask_logit.softmax(dim = 1)
            mask_pred = mask_prob.argmax(dim = 1)
            y_pred_mask.append(mask_pred.detach().cpu())

    y_pred_mask = torch.cat(y_pred_mask) 
    return y_pred_mask

# Test images
IMAGE_PATH_LIST_TEST = sorted(list(Path(image_path_test).glob("*.png")))


print(f'Total Images test: {len(IMAGE_PATH_LIST_TEST)}')

data_test = pd.DataFrame({'Image':IMAGE_PATH_LIST_TEST})
data_test.head()

# Dataset
class CustomTestDataset(Dataset):
    def __init__(self, data:pd.DataFrame, image_transforms):
        self.data = data
        self.image_transforms = image_transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)
        
        return image


test_dataset = CustomTestDataset(data_test, transforms_val_test)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)


# Run predictions
y_pred_mask = predictions_mask(test_dataloader)

# Visualizing the first 10 images of the test set and their predicted masks
IMAGE_PATH_LIST_TEST, MASK_PATH_LIST_TEST = image_mask_path(image_path_test, mask_path_test)

fig, ax = plt.subplots(nrows = 10
                       , ncols = 2 
                       , figsize = (12,35))

for index, row in data_test.iterrows():
    if index > 9:
        break
        
    img_bgr = cv2.imread(str(row[0]))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax[index, 0].imshow(img_rgb)
    ax[index, 0].axis('off')

    mask_path = row['Image'].name

    parts = str(mask_path).split("/")[-1].split("_")
    target_part = parts[0]
    target_parts = target_part.split("-")
    name_testset = "-".join(target_parts[1:-1] + ["_".join(parts[-3:-1])])

    ax[index, 0].set_title(f"Image: {name_testset}", fontsize=12, fontweight="bold", color="black")

    mask = y_pred_mask[index].squeeze().numpy()
    im = ax[index, 1].imshow(mask, cmap="viridis")  # Apply colormap

    ax[index, 1].axis('off')
    ax[index, 1].set_title("Mask", fontsize = 12, fontweight = "bold", color = "black")

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
fig.colorbar(im, cax=cbar_ax)

fig.tight_layout()
# fig.savefig(f"/home/wooju.chung/bmen619/figure/other_test_{job_id}.png")
plt.close(fig)

# Get image and mask paths
IMAGE_PATH_LIST_TEST, MASK_PATH_LIST_TEST = image_mask_path(image_path_test, 
                                                              mask_path_test)

fig, ax = plt.subplots(nrows = 10, ncols = 2, figsize = (20,30))

for i,(img_path, mask_path) in enumerate(zip(IMAGE_PATH_LIST_TEST, MASK_PATH_LIST_TEST)):
    if i>9:
        break
        
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    parts = str(mask_path).split("/")[-1].split("_")
    target_part = parts[0]
    target_parts = target_part.split("-")
    name_testset = "-".join(target_parts[1:-1] + ["_".join(parts[-3:-1])])  

    ax[i,0].imshow(img_rgb)
    ax[i,0].axis('off')
    ax[i,0].set_title(f"Image\nShape: {img_rgb.shape},{name_testset}", fontsize = 10, fontweight = "bold", color = "black")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ax[i,1].imshow(mask)
    ax[i,1].axis('off')
    ax[i,1].set_title(f"Mask\nShape: {mask.shape}", fontsize = 10, fontweight = "bold", color = "black")

fig.tight_layout()
# fig.savefig(f"/home/wooju.chung/bmen619/figure/other_groundtruth_{job_id}.png")
plt.close(fig)

# Pixel Accuracy
def pixel_accuracy(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    correct = torch.sum(pred == target)
    total = target.numel()
    return correct / total

# Dice Score
def dice(pred, target, num_classes, smooth=1e-6):
    dice_list = []
    for i in range(num_classes):
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        intersection = torch.sum(pred_i * target_i)
        union = torch.sum(pred_i) + torch.sum(target_i)
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_list.append(dice_score)
    return torch.mean(torch.stack(dice_list))  # 클래스별 평균 Dice Score 반환

# Mean IoU
def mean_iou(pred, target, num_classes):
    iou_list = []
    for i in range(num_classes):
        pred_class = (pred == i).float()
        target_class = (target == i).float()
        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(target_class) - intersection
        iou = intersection / (union + 1e-6)
        iou_list.append(iou)
    
    return torch.mean(torch.tensor(iou_list))

# Pixel Precision
def pixel_precision(pred, target, num_classes):
    precision_list = []
    for i in range(num_classes):
        pred_class = (pred == i).float()
        target_class = (target == i).float()
        true_positive = torch.sum(pred_class * target_class)
        false_positive = torch.sum(pred_class * (1 - target_class))
        precision = true_positive / (true_positive + false_positive + 1e-6)
        precision_list.append(precision)
    
    return torch.mean(torch.stack(precision_list))

# Pixel Recall
def pixel_recall(pred, target, num_classes):
    recall_list = []
    for i in range(num_classes):
        pred_class = (pred == i).float()
        target_class = (target == i).float()
        true_positive = torch.sum(pred_class * target_class)
        false_negative = torch.sum((1 - pred_class) * target_class)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        recall_list.append(recall)

    return torch.mean(torch.stack(recall_list))  # Return average Recall

# Pixel F1 Score
def pixel_f1_score(pred, target, num_classes):
    precision = pixel_precision(pred, target, num_classes)
    recall = pixel_recall(pred, target, num_classes)
    return 2 * (precision * recall) / (precision + recall + 1e-6)

# Function to evaluate metrics
def evaluate_metrics(pred, target, num_classes=models_number):
    pixel_acc = pixel_accuracy(pred, target)
    miou_val = mean_iou(pred, target, num_classes)
    dice_val_new = dice(pred, target, num_classes)
    precision_val_new = pixel_precision(pred, target, num_classes)
    recall_val_new = pixel_recall(pred, target, num_classes)
    f1_score_val_new = pixel_f1_score(pred, target, num_classes)
    
    return (pixel_acc, miou_val, dice_val_new, precision_val_new, recall_val_new, f1_score_val_new)


results = []

# Compare predictions with the target masks for batches
for idx in range(len(y_pred_mask)):
    pred = y_pred_mask[idx]
    mask_path = MASK_PATH_LIST_TEST[idx]

    parts = str(mask_path).split("/")[-1].split("_")
    target_part = parts[0]
    target_parts = target_part.split("-")
    name_testset = "-".join(target_parts[1:-1] + ["_".join(parts[-3:-1])])  

    target = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    target = torch.tensor(target, dtype=torch.int32)

    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    pixel_acc, miou_val, dice_val_new, precision_val_new, recall_val_new, f1_score_val_new = evaluate_metrics(pred, target)

    mask_path = str(mask_path).split("/")[-1]
    
    results.append({
        "image_name": mask_path,
        "pixel_accuracy": pixel_acc.item(),
        "dice_coefficient": dice_val_new.item(),
        "mean_iou": miou_val.item(),
        "pixel_precision": precision_val_new.item(),
        "pixel_recall": recall_val_new.item(),
        "pixel_f1_score": f1_score_val_new.item()
    })

df_results = pd.DataFrame(results)

# Optionally save the results to a CSV file
# df_results.to_csv(f'/home/wooju.chung/bmen619/metrics/evaluation_results_{job_id}.csv', index=False)
print('csv saved')