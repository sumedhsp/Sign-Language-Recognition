import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))

from custom_model_vision_v1 import SignLanguageRecognitionModelViT, I3DFeatureExtractor  # Ensure your model script is imported

def calculate_accuracy(outputs, labels):
    # Get the index of the max log-probability
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy


def run(configs, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', save_model='', weights=None):


    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=3,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=1,
                                                 pin_memory=False)
    
    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}


    i3d = InceptionI3d(100, in_channels=3)
    i3d.load_state_dict(torch.load('new_pretrained_100.pt', weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    num_classes = dataset.num_classes

    model = SignLanguageRecognitionModelViT(feature_extractor, num_classes)

    for param in model.feature_extractor.feature_extractor.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model.to(device)
    model = nn.DataParallel(model)


    lr = 1e-4
    weight_decay = 1e-5  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.module.transformer.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 100
    patience = 5  # For early stopping

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        total_batches = 0

        for batch_idx, (inputs, labels, vids) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(outputs, labels)
            running_loss += loss.item()
            running_accuracy += accuracy
            total_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloaders['train'])}], "
                    f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

        epoch_loss = running_loss / total_batches
        epoch_accuracy = running_accuracy / total_batches

        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, "
            f"Training Accuracy: {epoch_accuracy:.2f}%")


        if (epoch % 10 == 0):
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_accuracy = 0.0
            val_total_batches = 0

            with torch.no_grad():
                for val_batch_idx, (val_inputs, val_labels, val_vids) in enumerate(dataloaders['test']):
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    val_accuracy = calculate_accuracy(val_outputs, val_labels)

                    val_running_loss += val_loss.item()
                    val_running_accuracy += val_accuracy
                    val_total_batches += 1

            val_epoch_loss = val_running_loss / val_total_batches
            val_epoch_accuracy = val_running_accuracy / val_total_batches

            print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_epoch_loss:.4f}, "
                f"Validation Accuracy: {val_epoch_accuracy:.2f}%\n")

            # Scheduler step
            scheduler.step()

            # Check for improvement
            if val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = val_epoch_accuracy
                epochs_no_improve = 0

                # Save the best model
                checkpoint_path = os.path.join(checkpoint_dir, f"vision_best_model_{epoch}_{val_epoch_accuracy:.0f}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Validation accuracy improved. Model saved to {checkpoint_path}\n")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).\n")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    early_stop = True
                    break


    if not early_stop:
        # Save the final model
        final_model_path = os.path.join(checkpoint_dir, 'final_model_vision.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")


if __name__ == '__main__':
    mode = 'rgb'
    root = {'word': '../../data/WLASL2000'}
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_100.json'
    weights = None
    config_file = 'configfiles/asl100_vision.ini'

    configs = Config(config_file)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)













