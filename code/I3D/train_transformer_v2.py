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

from custom_models2 import SignLanguageRecognitionModelVision, I3DFeatureExtractor  # Ensure your model script is imported

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(configs, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', save_model='', weights=None):
    print(configs)

    # Setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224), videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    print ("Before dataset loading")
    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    print (val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test' : val_dataloader}

    print ("Before model setup")
    # Setup the model
    i3d = InceptionI3d(num_classes=100, in_channels=3)
    i3d.load_state_dict(torch.load('pre_trained_100.pt', weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    num_classes = dataset.num_classes
    model = SignLanguageRecognitionModelVision(feature_extractor, num_classes)

    if weights:
        print(f'Loading weights: {weights}')
        model.load_state_dict(torch.load(weights))

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    model = model.cuda()
    model = nn.DataParallel(model)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # Accumulate gradients
    steps = 0
    epoch = 0
    best_val_score = 0

    maxEpochs = 400
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < configs.max_steps and epoch < maxEpochs:  # Training loop
        print(f'Epoch {epoch}/{maxEpochs}')
        print(f'Step {steps}/{configs.max_steps}')
        print('-' * 10)

        epoch += 1
        for phase in ['train', 'test']:  # Add 'test' for evaluation phase
            if phase == 'train':
                model.train()
            else:
                model.eval()

            tot_loss = 0.0
            correct = 0
            total = 0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data
            for inputs, labels, _ in dataloaders[phase]:
                num_iter += 1
                inputs = inputs.cuda()

                # Process labels
                #print (labels.shape, labels)
                labels = labels.cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)  # Model predictions
                    #print ("Logits", logits)
                    #print ("Labels", labels)
                    loss = F.cross_entropy(logits, labels)

                    # Update metrics
                    tot_loss += loss.item()
                    _, preds = torch.max(logits, 1)  # Predicted class indices
                    correct += torch.sum(preds == labels).item()
                    total += labels.size(0)

                    if phase == 'train':
                        loss.backward()
                        if num_iter == num_steps_per_update:
                            steps += 1
                            optimizer.step()
                            optimizer.zero_grad()

            # Calculate average loss and accuracy
            epoch_loss = tot_loss / num_iter
            epoch_acc = correct / total

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test':
                val_score = tot_loss / num_iter
                if val_score > best_val_score:
                    best_val_score = val_score
                    model_name = f'{save_model}best_model_{steps:06d}.pt'
                    if not os.path.exists(save_model):
                        os.mkdir(save_model)

                    torch.save(model.module.state_dict(), model_name)
                    print(f'Model saved as {model_name}')

        scheduler.step(tot_loss / num_iter)


if __name__ == '__main__':
    mode = 'rgb'
    root = {'word': '../../data/WLASL2000'}
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_100.json'
    weights = None
    config_file = 'configfiles/asl100.ini'

    configs = Config(config_file)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
