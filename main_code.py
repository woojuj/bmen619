# Data handling
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
import cv2

# Torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from torchinfo import summary

# os
import os
# Path
from pathlib import Path

# tqdm
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

# job_id for comparing other results
import sys
if len(sys.argv) > 1:  # sys.argv[1]이 존재하는 경우
    job_id = sys.argv[1]
else:
    job_id = 0

# # **3. Load data**

# # 512 for test. less amount of patches
# image_path_train = '/home/wooju.chung/BCSS dataset/BCSS_512/smaller/train/'
# mask_path_train = '/home/wooju.chung/BCSS dataset/BCSS_512/smaller/train_mask/'

# image_path_val = '/home/wooju.chung/BCSS dataset/BCSS_512/smaller/val/'
# mask_path_val = '/home/wooju.chung/BCSS dataset/BCSS_512/smaller/val_mask/'

# image_path_test = '/home/wooju.chung/BCSS dataset/BCSS_512/smaller/test/'
# TEST_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/smaller/test_mask/'

# # 224 for test. less amount of patches
# image_path_train = '/home/wooju.chung/BCSS dataset/BCSS/smaller/train/'
# mask_path_train = '/home/wooju.chung/BCSS dataset/BCSS/smaller/train_mask/'

# image_path_val = '/home/wooju.chung/BCSS dataset/BCSS/smaller/val/'
# mask_path_val = '/home/wooju.chung/BCSS dataset/BCSS/smaller/val_mask/'

# image_path_test = '/home/wooju.chung/BCSS dataset/BCSS/smaller/test/'
# TEST_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS/smaller/test_mask/'

# 전체 512
image_path_train = '/home/wooju.chung/BCSS dataset/BCSS_512/train_512/'
mask_path_train = '/home/wooju.chung/BCSS dataset/BCSS_512/train_mask_512/'

image_path_val = '/home/wooju.chung/BCSS dataset/BCSS_512/val_512/'
mask_path_val = '/home/wooju.chung/BCSS dataset/BCSS_512/val_mask_512/'

image_path_test = '/home/wooju.chung/BCSS dataset/BCSS_512/test_512/'
TEST_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/test_mask_512/'

# 전체 224
image_path_train = '/home/wooju.chung/BCSS dataset/BCSS/train/'
mask_path_train = '/home/wooju.chung/BCSS dataset/BCSS/train_mask/'

image_path_val = '/home/wooju.chung/BCSS dataset/BCSS/val/'
mask_path_val = '/home/wooju.chung/BCSS dataset/BCSS/val_mask/'

image_path_test = '/home/wooju.chung/BCSS dataset/BCSS/test/'
TEST_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS/test_mask/'

# When changing models, set the number of out channels
models_number= 3
#22 is 512, 3 is 224

# We define a function to create a list of the paths of the images and masks.
def image_mask_path(image_path:str, mask_path:str):
    IMAGE_PATH = Path(image_path)
    IMAGE_PATH_LIST = sorted(list(IMAGE_PATH.glob("*.png")))

    MASK_PATH = Path(mask_path)
    MASK_PATH_LIST = sorted(list(MASK_PATH.glob("*.png")))
    
    return IMAGE_PATH_LIST, MASK_PATH_LIST

# # 
# # image_path_train = "/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS_512/train_512"
# image_path_train = "/Users/wj/Downloads/BMEN619.03/Project/BCSS dataset/BCSS_512/train_512"
# # mask_path_train = "/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS_512/train_mask_512"
# mask_path_train = "/Users/wj/Downloads/BMEN619.03/Project/BCSS dataset/BCSS_512/train_mask_512"

# # 
# # image_path_val = "/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS_512/val_512"
# # mask_path_val = "/kaggle/input/breast-cancer-semantic-segmentation-bcss/BCSS_512/val_mask_512"
# image_path_val = "/Users/wj/Downloads/BMEN619.03/Project/BCSS dataset/BCSS_512/val_512"
# mask_path_val = "/Users/wj/Downloads/BMEN619.03/Project/BCSS dataset/BCSS_512/val_mask_512"

# # # 원본
# TRAIN_IMAGE_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/train_512/'
# VAL_IMAGE_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/val_512/'
# TRAIN_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/train_mask_512/'
# VAL_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/val_mask_512/'
# TEST_IMAGE_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/test_512/'
# TEST_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/test_mask_512/'


IMAGE_PATH_LIST_TRAIN, MASK_PATH_LIST_TRAIN = image_mask_path(image_path_train, 
                                                              mask_path_train)

print(f'Total Images Train: {len(IMAGE_PATH_LIST_TRAIN)}')
print(f'Total Masks Train: {len(MASK_PATH_LIST_TRAIN)}')

IMAGE_PATH_LIST_VAL, MASK_PATH_LIST_VAL = image_mask_path(image_path_val, 
                                                          mask_path_val)

print(f'Total Images Val: {len(IMAGE_PATH_LIST_VAL)}')
print(f'Total Masks Val: {len(MASK_PATH_LIST_VAL)}')

# 
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

# 
# 왜 마스크색이 다 다르지? imshow의 내장기능이었다..
# We see some images and next to them their respective mask.
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
    # ax[i,1].imshow(mask, cmap='gray')
    ax[i,1].axis('off')
    ax[i,1].set_title(f"Mask\nShape: {mask.shape}", fontsize = 10, fontweight = "bold", color = "black")

fig.tight_layout()
fig.savefig(f"/home/wooju.chung/bmen619/figure/other_sample_{job_id}.png")
plt.close(fig)

# We visualize some images but with the mask superimposed.
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
fig.savefig(f"/home/wooju.chung/bmen619/figure/other_mask_superimposed_{job_id}.png")
plt.close(fig)

#  [markdown]
# # **4. Preprocessing**

#  [markdown]
# **We will create dataframes for both data sets.**

# 
data_train = pd.DataFrame({'Image':IMAGE_PATH_LIST_TRAIN, 
                           'Mask':MASK_PATH_LIST_TRAIN})

data_val = pd.DataFrame({'Image':IMAGE_PATH_LIST_VAL, 
                         'Mask':MASK_PATH_LIST_VAL})

#  [markdown]
# **Now we are going to find out what transformations were applied to the images when the model was pre-trained in order to replicate it in our images.**

# 
preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name = "resnet34", 
                                        pretrained = "imagenet")
# preprocess_input

#  [markdown]
# **We are going to replicate this same thing.**

# 
RESIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

image_transforms = transforms.Compose([transforms.Resize(RESIZE),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = MEAN, std = STD)])

mask_transforms = transforms.Compose([transforms.Resize(RESIZE), 
                                      transforms.PILToTensor()])

#  [markdown]
# **We define our Dataset with all the transformations to perform.**

#  [markdown]
# - **Dataset**

# 
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

# 
# 여기에서 멀티프로세싱을 위한 설정을 합니다

# import torch.multiprocessing as mp

# # 여기에서 멀티프로세싱을 위한 설정을 합니다
# mp.set_start_method('spawn', force=True)

# if __name__ == "__main__":
train_dataset = CustomImageMaskDataset(data_train, image_transforms, 
                                    mask_transforms)
    
val_dataset = CustomImageMaskDataset(data_val, image_transforms, 
                                    mask_transforms)
# BATCH_SIZE = 64
# NUM_WORKERS = os.cpu_count() #에러나서 잠시 0으로
# # NUM_WORKERS= 0

# train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, 
#                             shuffle = True, num_workers = NUM_WORKERS)

# val_dataloader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, 
#                             shuffle = True, num_workers = NUM_WORKERS)

#  [markdown]
# - **DataLoader**

# if __name__ == "__main__":
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count() #에러나서 잠시 0으로
print(NUM_WORKERS, ': NUM_WORKERS')
# NUM_WORKERS= 0

train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, 
                            shuffle = True, num_workers = NUM_WORKERS)

val_dataloader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, 
                            shuffle = True, num_workers = NUM_WORKERS)


# 
# We visualize the dimensions of a batch.
batch_images, batch_masks = next(iter(train_dataloader))

batch_images.shape, batch_masks.shape

# 
# We visualize the dimensions of a batch.
batch_images, batch_masks = next(iter(train_dataloader))

batch_images.shape, batch_masks.shape # 단순확인용

#  [markdown]
# # **5. Model**

# 
# CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# 
# Define model
model = smp.Unet(in_channels = 3, classes = models_number)
# 이거만 바꾸면되나? 224

# 서머리 걍 보는용같아서 주석함 하나 더있음 밑에
# Now Let's visualize the architecture of our model.
# summary(model = model, 
#         input_size = [64, 3, 224, 224], 
#         col_width = 15, 
#         col_names = ['input_size', 'output_size', 'num_params', 'trainable'], 
#         row_settings = ['var_names'])

#  [markdown]
# **Because we are going to use transfer learning we are going to freeze the encoder layer.**

# 
for param in model.encoder.parameters():
    param.requires_grad = False

# ## 
# # We view our model again to check if the encoder layers freeze.
# summary(model = model, 
#         input_size = [64, 3, 224, 224], 
#         col_width = 15, 
#         col_names = ['input_size', 'output_size', 'num_params', 'trainable'], 
#         row_settings = ['var_names'])

#  [markdown]
# **Great !!, now we have to define the `loss function` and the `optimizer`.**

# 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)

#  [markdown]
# **Now we move on to the training stage, for this we are going to define some functions to execute the training and the final predictions.**
# 
# - **`EarlyStopping`**
# - **`train_step`**
# - **`val_step`**
# - **`predictions_mask`**

# 
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

# 
# Define early stopping

early_stopping = EarlyStopping(patience = 20, delta = 0.)

# 
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
                                            num_classes = models_number) # 224때 변경?
        
        train_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction = "micro").numpy()
        
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy

# 
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

# 
def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, 
          val_dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, 
          optimizer:torch.optim.Optimizer, early_stopping, epochs:int = 10):
    
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
        
        if early_stopping.early_stop == True:
            print("Early Stopping!!!")
            break
            
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_accuracy)
        
    return results


# Training!!!

SEED = 42
EPOCHS = 4
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

RESULTS = train(model.to(device = DEVICE), 
                train_dataloader, 
                val_dataloader, 
                loss_fn, 
                optimizer, 
                early_stopping, 
                EPOCHS)

#  [markdown]
# # **6. Metrics**

# 
# We define a function to visualize the evolution of the loss and the metric.
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
    
    ax[1].plot(training_metric, label = "Train")
    ax[1].plot(validation_metric, label = "Val")
    ax[1].set_title("Accuracy", fontsize = 12, fontweight = "bold", color = "black")
    ax[1].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[1].set_ylabel("score", fontsize = 10, fontweight = "bold", color = "black")
    
    fig.tight_layout()
    fig.savefig(f"/home/wooju.chung/bmen619/figure/other_loss_acc_{job_id}.png")
    plt.close(fig)

# 
loss_and_metric_plot(RESULTS)

#  [markdown]
# # **7. Final Predictions**

# 
def predictions_mask(test_dataloader:torch.utils.data.DataLoader):
    
    checkpoint = torch.load(f"/home/wooju.chung/bmen619/model/best_model_{job_id}.pth")

    loaded_model = smp.Unet(encoder_weights = None, classes = models_number) # 224 여기도?

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

#  [markdown]
# **We are going to perform all the previous steps that we do to transform our data and get it ready to enter into the model.**


#테스트용 이미지
IMAGE_PATH_LIST_TEST = list(Path(image_path_test).glob("*.png"))

print(f'Total Images test: {len(IMAGE_PATH_LIST_TEST)}')

# 
data_test = pd.DataFrame({'Image':IMAGE_PATH_LIST_TEST})
data_test.head()

#  [markdown]
# - **Dataset**

# 
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

# 
# Dataset
test_dataset = CustomTestDataset(data_test, image_transforms)

# DataLoader
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# 
# We execute the predictions!!
y_pred_mask = predictions_mask(test_dataloader)

#  [markdown]
# **We visualize the first 10 images of our test set and the predicted mask.**
IMAGE_PATH_LIST_TEST, MASK_PATH_LIST_TEST = image_mask_path(image_path_test, 
                                                              TEST_MASK_PATH)
# 
fig, ax = plt.subplots(nrows = 10
                       , ncols = 2 # 2
                       , figsize = (12,35)) #12->18

for index, row in data_test.iterrows():
    if index > 9:
        break
        
    img_bgr = cv2.imread(str(row[0]))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax[index, 0].imshow(img_rgb)
    ax[index, 0].axis('off')
    ax[index, 0].set_title("Image", fontsize = 12, fontweight = "bold", color = "black")
    
    ax[index, 1].imshow(y_pred_mask[index].squeeze().numpy())
    ax[index, 1].axis('off')
    ax[index, 1].set_title("Mask", fontsize = 12, fontweight = "bold", color = "black")
    
##### 보여주는거 하다가 지침
# # 경로 설정
# TEST_MASK_PATH = '/home/wooju.chung/BCSS dataset/BCSS_512/smaller/test_mask/'

# # n개의 이미지를 불러올 개수 설정
# n = 5  # 예시로 5개 이미지를 불러오겠습니다.

# # 이미지 파일 리스트 가져오기
# image_files = [f for f in os.listdir(TEST_MASK_PATH) if f.endswith(('.png'))]

# # 최대 n개의 이미지를 불러오기
# image_files = image_files[:n]

# for i, ax_row in enumerate(ax):
#     print()
#     img_path = os.path.join(TEST_MASK_PATH, image_files[i])
#     img = Image.open(img_path)
#     ax_row[2].imshow(img)
#     ax_row[2].axis('off')
#     ax_row[2].set_title(f"Image {i + 1}", fontsize=12, fontweight="bold", color="black")
##### 보여주는거 하다가 지침

fig.tight_layout()
fig.savefig(f"/home/wooju.chung/bmen619/figure/other_test_{job_id}.png")
plt.close(fig)

# #정답확인 이건됨

IMAGE_PATH_LIST_TEST, MASK_PATH_LIST_TEST = image_mask_path(image_path_test, 
                                                              TEST_MASK_PATH)

fig, ax = plt.subplots(nrows = 10, ncols = 2, figsize = (20,30))

for i,(img_path, mask_path) in enumerate(zip(IMAGE_PATH_LIST_TEST, MASK_PATH_LIST_TEST)):
    if i>9:
        break
        
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax[i,0].imshow(img_rgb)
    ax[i,0].axis('off')
    ax[i,0].set_title(f"Image\nShape: {img_rgb.shape}", fontsize = 10, fontweight = "bold", color = "black")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ax[i,1].imshow(mask)
    # ax[i,1].imshow(mask, cmap='gray')
    ax[i,1].axis('off')
    ax[i,1].set_title(f"Mask\nShape: {mask.shape}", fontsize = 10, fontweight = "bold", color = "black")

fig.tight_layout()
fig.savefig(f"/home/wooju.chung/bmen619/figure/other_groundtruth_{job_id}.png")
plt.close(fig)