import math
import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms

import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

# from nslt_dataset_all import NSLT as Dataset
from datasets.nslt_dataset_all import NSLT as Dataset

from custom_models import SignLanguageRecognitionModel, I3DFeatureExtractor  # Ensure your model script is imported

import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))



def compute_topk_tp_fp(outputs, labels, num_classes, topk=(1, 5, 10)):
    """
    Computes True Positives and False Positives per class for specified Top-K.

    Args:
        outputs (torch.Tensor): Model outputs (logits) of shape [num_samples, num_classes].
        labels (torch.Tensor): Ground truth labels of shape [num_samples].
        num_classes (int): Total number of classes.
        topk (tuple): Tuple of K values for Top-K metrics.

    Returns:
        dict: Nested dictionary with K as keys and dictionaries of TP and FP per class.
    """
    # Initialize dictionaries to hold TP and FP counts per class for each K
    topk_metrics = {k: {'TP': defaultdict(int), 'FP': defaultdict(int)} for k in topk}
    
    # Move tensors to CPU and convert to numpy for processing
    outputs_np = outputs.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Get the indices of the top K predictions for each sample
    topk_preds = {}
    for k in topk:
        # Argsort in descending order and take the first k
        topk_preds[k] = np.argsort(outputs_np, axis=1)[:, -k:][:, ::-1]  # Shape: [num_samples, k]
    
    # Iterate over each sample to compute TP and FP
    for i in range(len(labels_np)):
        true_label = labels_np[i]
        for k in topk:
            preds_k = topk_preds[k][i]
            if true_label in preds_k:
                topk_metrics[k]['TP'][true_label] += 1
            # False Positives: Increment FP for all predicted classes except the true label
            for pred in preds_k:
                if pred != true_label:
                    topk_metrics[k]['FP'][pred] += 1
    
    return topk_metrics

def compute_per_class_topk_accuracy(topk_metrics, num_classes, topk):
    """
    Computes per-class Top-K average accuracy based on TP and FP.

    Args:
        topk_metrics (dict): Nested dictionary with K as keys and dictionaries of TP and FP per class.
        num_classes (int): Total number of classes.
        topk (tuple): Tuple of K values.

    Returns:
        dict: Nested dictionary with K as keys and per-class accuracy dictionaries.
    """
    per_class_accuracy = {k: {} for k in topk}
    
    for k in topk:
        for cls in range(num_classes):
            TP = topk_metrics[k]['TP'][cls]
            FP = topk_metrics[k]['FP'][cls]
            # To calculate per-class accuracy, you might consider:
            # Precision = TP / (TP + FP)
            # Recall = TP / (Actual Positives)
            # However, since you want average per-class accuracy, define accordingly.
            # Here, we'll compute Precision as a representative metric.
            if TP + FP > 0:
                precision = TP / (TP + FP)
            else:
                precision = 0.0  # No predictions for this class
            
            per_class_accuracy[k][cls] = precision
    
    return per_class_accuracy


def run(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    
    # setup the model
    i3d = InceptionI3d(1000, in_channels=3)
    i3d.load_state_dict(torch.load('i3d_pretrained_1000.pt', weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    num_classes = val_dataset.num_classes

    model = SignLanguageRecognitionModel(feature_extractor, num_classes)
    
    model.cuda()
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load("test_vanilla_1000.pth", weights_only=True))

    model.eval()

    all_outputs = []
    all_labels = []
    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        outputs = model(inputs)

        
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    outputs_np = all_outputs.detach().cpu().numpy()
    labels_np = all_labels.cpu().numpy()

    # Printing the labels and outputs to validate in local machine
    
    np.savetxt('outputs.txt', outputs_np, delimiter=',')
    np.savetxt('labels.txt', labels_np, delimiter=',')

    topk = (1, 5, 10)
    topkmetrics = compute_topk_tp_fp(all_outputs, all_labels, num_classes, topk)


    # Compute per-class Top-K average accuracy (Precision)
    #per_class_accuracy = compute_per_class_topk_accuracy(topkmetrics, num_classes, topk)

    # Display the results
    #for k in topk:
    #    print(f"\nTop-{k} Per-Class Accuracy (Precision):")
    #    for cls in range(num_classes):
    #        print(f"Class {cls}: {per_class_accuracy[k][cls]*100:.2f}%")


    _, preds = torch.max(all_outputs, 1)
    all_preds = preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("Confusion Matrix:")
    np.save('confusionMatrix.npy', confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    # ================== test i3d on a dataset ==============
    # need to add argparse
    mode = 'rgb'
    num_classes = 1000
    save_model = './checkpoints/'

    root = '../../data/WLASL2000'

    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
#     weights = 'test_vanilla_v1.pth'

    run(mode=mode, root=root, save_model=save_model, train_split=train_split )
