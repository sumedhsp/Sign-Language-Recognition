import json
import math
import os
import random
import time
import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utl
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))

# --- Dataset and Utility Functions (Refactored NSLT Class) ---

def video_to_tensor(pic):
    if isinstance(pic, np.ndarray):
        # If pic is a NumPy array, transpose it
        return torch.from_numpy(pic.transpose([3, 0, 1, 2]))
    elif isinstance(pic, torch.Tensor):
        # If pic is already a tensor, ensure it has the correct shape
        return pic.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
    else:
        raise TypeError(f"Unsupported type for pic: {type(pic)}. Expected np.ndarray or torch.Tensor.")

    #return torch.from_numpy(pic.transpose(3, 0, 1, 2))


def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(224, 224)):
    video_path = os.path.join(vid_root, vid + '.mp4')

    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, total_frames - start)):
        success, img = vidcap.read()
        if not success:
            break  # Stop if no more frames

        # Resize if necessary
        h, w, c = img.shape
        if h < 226 or w < 226:
            d = 226.0 - min(h, w)
            sc = 1 + d / min(h, w)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        if h > resize[1] or w > resize[0]:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)

        # Normalize to [-1, 1]
        img = (img.astype(np.float32) / 255.0) * 2.0 - 1.0

        frames.append(img)

    vidcap.release()

    if len(frames) < num:
        # Pad by repeating the last frame
        pad_length = num - len(frames)
        if pad_length > 0:
            pad_frame = frames[-1]
            frames.extend([pad_frame] * pad_length)


    if len(frames) == 0:
        print(f"Warning: No frames loaded for video {video_path}")

    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    count_skipping = 0
    for vid in data.keys():
        # Split handling
        if split == 'train':
            if data[vid]['subset'] not in ['train', 'val']:
                continue
        else:
            if data[vid]['subset'] != 'test':
                continue

        vid_root = root['word']
        video_path = os.path.join(vid_root, vid + '.mp4')
        if not os.path.exists(video_path):
            continue

        # Get total frames
        vidcap = cv2.VideoCapture(video_path)
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()

        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames < 9:
            print(f"Skip video {vid} due to insufficient frames: {num_frames}")
            count_skipping += 1
            continue

        class_id = data[vid]['action'][0]
        label = class_id

        # Handle different video ID lengths
        if len(vid) == 5:
            dataset.append((vid, label, 0, 0, num_frames))
        elif len(vid) == 6:  # sign kws instances
            start_frame = data[vid]['action'][1]
            duration = data[vid]['action'][2] - data[vid]['action'][1]
            dataset.append((vid, label, 0, start_frame, duration))

    print(f"Skipped videos: {count_skipping}")
    print(f"Total videos in dataset: {len(dataset)}")
    return dataset


def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))
    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)


class WLASLDataset(data_utl.Dataset):
    def __init__(self, split_file, split, root, mode, transforms=None, num_frames=64):
        self.num_classes = get_num_class(split_file)
        self.data = make_dataset(split_file, split, root, mode, num_classes=self.num_classes)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.num_frames = num_frames
        self.label_encoder = self._get_label_encoder()

    def _get_label_encoder(self):
        # Initialize label encoder
        labels = [item[1] for item in self.data]
        le = LabelEncoder()
        le.fit(labels)
        return le

    def __getitem__(self, index):
        vid, label, src, start_frame, nf = self.data[index]

        total_frames = 64
        # Sample starting frame ensuring we have enough frames
        try:
            start_f = random.randint(start_frame, nf - self.num_frames - 1 + start_frame)
        except ValueError:
            start_f = start_frame

        # Load frames
        imgs = load_rgb_frames_from_video(self.root['word'], vid, start_f, self.num_frames)

        # Apply transforms
        #if self.transforms:
        #    imgs = self.transforms(imgs)  # Expected shape: [num_frames, H, W, C]

        # Debugging: Check type and shape of imgs
            # Pad if fewer frames
        if imgs.shape[0] < self.num_frames:
            num_padding = self.num_frames - imgs.shape[0]
            pad_frame = imgs[-1]  # Use last frame for padding
            padding = np.tile(np.expand_dims(pad_frame, axis=0), (num_padding, 1, 1, 1))
            imgs = np.concatenate([imgs, padding], axis=0)

        # Debugging: Check shape after padding
        #print(f"Final imgs shape after padding and resizing: {imgs.shape}")

        # Convert to tensor and permute to [C, T, H, W]
        imgs = video_to_tensor(imgs)  # [C, T, H, W]

        # Encode label
        label_encoded = self.label_encoder.transform([label])[0]
        ret_lab = torch.tensor(label_encoded, dtype=torch.long)
        ret_img = imgs

        return ret_img, ret_lab, vid

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                else:
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs

        #label = label[:, 0]
        #label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label

# --- Data Transformations ---

class VideoTransforms:
    def __init__(self, resize=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, video):
        # video: [num_frames, H, W, C] in range [-1, 1]
        # Convert to [C, T, H, W]
        video = torch.from_numpy(video.transpose(3, 0, 1, 2))  # [C, T, H, W]
        # Apply normalization
        video = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(video)
        return video

class VideoAugTransforms:
    def __init__(self, resize=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, video):
        # video: [num_frames, H, W, C] in range [-1, 1]
        # Convert to [C, T, H, W]
        video = torch.from_numpy(video.transpose(3, 0, 1, 2))  # [C, T, H, W]
        
        # Apply transforms frame-wise
        C, T, H, W = video.shape
        transformed_frames = []
        for t in range(T):
            frame = video[:, t, :, :]  # [C, H, W]
            frame = self.transform(frame)  # Apply transformations
            transformed_frames.append(frame.unsqueeze(1))  # [C, 1, H, W]
        
        # Concatenate frames back
        video = torch.cat(transformed_frames, dim=1)  # [C, T, H, W]
        return video


# --- DataLoader Creation ---

def get_dataloaders(root_dir, split_file, batch_size=8, num_workers=4, num_frames=64):
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        if split == 'train':
            transform = VideoAugTransforms(resize=(224, 224))
        else:
            transform = VideoTransforms(resize=(224, 224))
        
        dataset = WLASLDataset(
            split_file=split_file,
            split=split,
            root=root_dir,
            mode='rgb',  # Assuming 'rgb' mode; adjust if needed
            transforms=transform,
            num_frames=num_frames
        )
        
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        dataloaders[split] = dataloader
    return dataloaders


# --- Model Definition ---

import torch
import torch.nn as nn
from torchvision import models

class ViTLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=2, pretrained=True):
        super(ViTLSTM, self).__init__()
        
        # Load pre-trained ViT
        self.vit = models.vit_b_16(pretrained=pretrained)
        # Remove the classification head
        self.vit.heads = nn.Identity()
        
        for param in self.vit.parameters():
            param.requires_grad = False  # Freeze all layers


        # Dynamically get embedding dimension
        if hasattr(self.vit, "hidden_dim"):
            self.embed_dim = self.vit.hidden_dim
        elif hasattr(self.vit, "embedding_dim"):
            self.embed_dim = self.vit.embedding_dim
        else:
            self.embed_dim = self.vit.heads.in_features  # Fallback
        
        # Define LSTM
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        with torch.no_grad():  # Avoid gradients for ViT
            frame_features = self.vit(x)  # [B*T, embed_dim]
        frame_features = frame_features.view(B, T, self.embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(frame_features)
        final_feature = h_n[-1]  # [B, hidden_dim]
        return self.classifier(final_feature)

        # x shape: [B, C, T, H, W]
        '''B, C, T, H, W = x.shape
        
        # Reshape to [B*T, C, H, W] to process all frames at once
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        
        # Extract frame features using ViT
        with torch.set_grad_enabled(self.training):
            frame_features = self.vit(x)  # [B*T, D]
        
        # Reshape to [B, T, D]
        frame_features = frame_features.view(B, T, -1)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(frame_features)  # lstm_out: [B, T, hidden_dim]
        
        # Use the last hidden state for classification
        final_feature = h_n[-1]  # [B, hidden_dim]
        out = self.classifier(final_feature)  # [B, num_classes]
        
        return out'''


def get_model(num_classes, pretrained=True):
    model = ViTLSTM(num_classes=num_classes, hidden_dim=256, num_layers=2, pretrained=True)

    return model

def customize_model(model, num_classes):
    # Replace the classification head
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features, num_classes)
    return model


# --- Training and Evaluation Functions ---

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels, vids in dataloaders[phase]:
                inputs = inputs.to(device)  # [C, T, H, W]
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # [batch_size, num_classes]
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed //60:.0f}m {time_elapsed %60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloader, device, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, vids in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
    print("Classification Report:\n", report)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("Confusion Matrix:\n", cm_df)


def freeze_layers(model, freeze_until=6):
    """
    Freezes layers in the model up to a specified transformer layer.

    Args:
        model (nn.Module): The TimeSformer model.
        freeze_until (int): The index of the transformer layer up to which layers are frozen.
    """
    for name, param in model.named_parameters():
        if 'blocks.' in name:
            layer_num = int(name.split('.')[1])
            if layer_num < freeze_until:
                param.requires_grad = False

if __name__ == "__main__":
    # Define dataset root and split file paths
    root_dir = {
        'word': '../../data/WLASL2000'  # Replace with your dataset path
    }
    split_file = 'nslt_100.json'  # Replace with your split file path
    
    # Create DataLoaders
    dataloaders = get_dataloaders(
        root_dir=root_dir,
        split_file=split_file,
        batch_size=16,
        num_workers=4,
        num_frames=64
    )

    # Get number of classes
    num_classes = get_num_class(split_file=split_file)
    print(f"Number of classes: {num_classes}")

    # Initialize the model
    model = get_model(num_classes=num_classes, pretrained=True)
    #model = customize_model(model, num_classes)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Define loss function, optimizer, and scheduler
    #freeze_layers(model, freeze_until=6)
    # Re-define optimizer to only update unfrozen parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # Optionally, define a different scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 25
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device
    )

    # Save the best model
    torch.save(trained_model.state_dict(), 'best_timesformer_wlasl.pth')
