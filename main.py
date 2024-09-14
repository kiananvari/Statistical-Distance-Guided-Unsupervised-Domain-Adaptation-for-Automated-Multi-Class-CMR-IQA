import os
import pickle
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from models import * 

from utils import AverageMeter, Logger
from center_loss import CenterLoss

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, transforms

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='cnn')
# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='plot_image')
parser.add_argument('--plot',  type=int, default=1)

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # print("Creating dataset: {}".format(args.dataset))
    # dataset = datasets.create(
    #     name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
    #     num_workers=args.workers,
    # )
    # num_classes = 10
    # trainloader, testloader = dataset.trainloader, dataset.testloader

    #################################################################

    with open('C:\\Users\\Vision\\Documents\\Users\\Kian\\IMP2+CL\\center_loss\\ACDC_IMAGE.pkl', 'rb') as f:
        final_dataset = pickle.load(f)
    print("CMRxRecon_IMAGE.pkl Dataset Loaded successfully!")

    src_train_dataset = final_dataset[0]
    src_test_dataset = final_dataset[1]

    with open('C:\\Users\\Vision\\Documents\\Users\\Kian\\IMP2+CL\\center_loss\\CMRxRecon_IMAGE.pkl', 'rb') as f:
        final_dataset = pickle.load(f)
    print("ACDC_IMAGE.pkl Dataset Loaded successfully!")

    tar_train_dataset = final_dataset[0]
    tar_test_dataset = final_dataset[1]

    #################################################################
    # test_datasets_names = ["test_Datas et_ACDC"]
    # test_datasets = [test_dataset]
    #################################################################

    # Prepare the dataset
    class MotionArtifactDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.transform = transforms.ToTensor()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            image, label = self.data[index]
            image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1
            image = image * 255  # Scale to 0-255 range
            # image = cv2.resize(image, (15, 102))
            image = self.transform(image).float()
            image = np.array(image)

            return image, label
    #################################################################
    # # Create data loaders for the current fold
    # batch_size = 256  # Adjust the batch size as needed
    # trainloader = DataLoader(MotionArtifactDataset(train_dataset), batch_size=batch_size, shuffle=True)
    # testloader = DataLoader(MotionArtifactDataset(test_dataset), batch_size=batch_size, shuffle=True)

    # num_classes = 5

    # Create data loaders for the current fold
    batch_size = 310  # Adjust the batch size as needed
    src_train_loader = DataLoader(MotionArtifactDataset(src_train_dataset), batch_size=batch_size, shuffle=True)
    tar_train_loader = DataLoader(MotionArtifactDataset(tar_train_dataset), batch_size=batch_size, shuffle=True)

    src_test_loader = DataLoader(MotionArtifactDataset(src_test_dataset), batch_size=batch_size, shuffle=True)
    tar_test_loader = DataLoader(MotionArtifactDataset(tar_test_dataset), batch_size=batch_size, shuffle=True)

    num_classes = 5
    #################################################################

    print("Creating model: {}".format(args.model))
    # model = models.create(name=args.model, num_classes=num_classes)
    extractor = Extractor()
    classifier = Classifier()
    discriminator = Discriminator_WGAN()

    if use_gpu:
        extractor = nn.DataParallel(extractor).cuda()
        classifier = nn.DataParallel(classifier).cuda()
        discriminator = nn.DataParallel(discriminator).cuda()

    class_criterion = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=use_gpu)


    c_opt = torch.optim.Adam([{"params": classifier.parameters()},
                              {"params": extractor.parameters()}], lr=1e-3, weight_decay=5e-04)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    optimizer_centloss = torch.optim.Adam(criterion_cent.parameters(), lr=args.lr_cent)


    # criterion_xent = nn.CrossEntropyLoss()
    # # optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    # optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=5e-04)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(c_opt, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()
    results = []

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(extractor, classifier, discriminator, class_criterion, criterion_cent, c_opt, d_opt, optimizer_centloss,
              src_train_loader, tar_train_loader, use_gpu, num_classes, epoch)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test ON SCR:")
            acc, err = test(extractor, classifier, src_test_loader, use_gpu, num_classes, epoch, results, type = "SRC")
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            print("==> Test ON TAR:")
            acc, err = test(extractor, classifier, tar_test_loader, use_gpu, num_classes, epoch, results, type = "TAR")
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(extractor, classifier, discriminator, class_criterion, criterion_cent, c_opt, d_opt, optimizer_centloss,
        src_train_loader, tar_train_loader, use_gpu, num_classes, epoch):
    
    extractor.train()
    classifier.train()

    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    wd_losses = AverageMeter()
    losses = AverageMeter()
    
    if args.plot:
        all_src_features, all_src_labels = [], []
        all_tar_features, all_tar_labels = [], []

    for batch_idx, (src, tar) in enumerate(zip(src_train_loader, tar_train_loader)):

        src_data, src_label = src
        tar_data, tar_label = tar

        if use_gpu:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

        """ train classifier """
        set_requires_grad(extractor, requires_grad=True)
        set_requires_grad(discriminator, requires_grad=False)

        c_opt.zero_grad()
        d_opt.zero_grad()
        optimizer_centloss.zero_grad()

        src_z = extractor(src_data)
        tar_z = extractor(tar_data)

        cent_feature, pred_class = classifier(src_z)

        class_loss = class_criterion(pred_class, src_label)

        loss_cent = criterion_cent(cent_feature, src_label)
        loss_cent *= args.weight_cent

        wasserstein_diatance = discriminator(src_z).mean() - discriminator(tar_z).mean()

        alpha = min(1.0, epoch / 200.0)
        loss = class_loss + loss_cent + (alpha * wasserstein_diatance)

        loss.backward()
        c_opt.step()

        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()

        
        losses.update(loss.item(), src_label.size(0))
        xent_losses.update(class_loss.item(), src_label.size(0))
        cent_losses.update(loss_cent.item(), src_label.size(0))
        wd_losses.update(wasserstein_diatance.item(), src_label.size(0))

        # with torch.no_grad():
        #     tar_features, tar_outputs = model(tar_data)

        if args.plot:
            if use_gpu:
                all_src_features.append(src_z.data.cpu().numpy())
                all_src_labels.append(src_label.data.cpu().numpy())

                # all_tar_features.append(tar_features.data.cpu().numpy())
            #     # all_tar_labels.append(tar_label.data.cpu().numpy())
            # else:
            #     all_src_features.append(src_features.data.numpy())
            #     all_src_labels.append(tar_label.data.numpy())

                # all_tar_features.append(tar_features.data.numpy())
                # all_tar_labels.append(tar_label.data.numpy())


        """ train discriminator """
        discriminator.train()
        set_requires_grad(extractor, requires_grad=False)
        set_requires_grad(discriminator, requires_grad=True)

        with torch.no_grad():
            src_z = extractor(src_data)
            tar_z = extractor(tar_data)

        for _ in range(10):
            gp = gradient_penalty(discriminator, src_z, tar_z)
            d_src_loss = discriminator(src_z)
            d_tar_loss = discriminator(tar_z)

            wasserstein_distance = d_src_loss.mean() - d_tar_loss.mean()

            domain_loss = -wasserstein_distance + 10*gp

            d_opt.zero_grad()
            domain_loss.backward()
            d_opt.step()


        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) ClassLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})  WD {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(src_train_loader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, cent_losses.val, cent_losses.avg, wd_losses.val, wd_losses.avg))

    if args.plot:
        all_src_features = np.concatenate(all_src_features, 0)
        all_src_labels = np.concatenate(all_src_labels, 0)

        # all_tar_features = np.concatenate(all_tar_features, 0)
        # all_tar_labels = np.concatenate(all_tar_labels, 0)

        plot_features(all_src_features, all_src_labels, num_classes, epoch, prefix='train_ON_SRC')
        # plot_features(all_tar_features, all_tar_labels, num_classes, epoch, prefix='train_ON_TAR')


def test(extractor, classifier, testloader, use_gpu, num_classes, epoch, results, type):
    
    extractor.eval()
    classifier.eval()

    test_correct = 0
    test_predictions = []
    test_true_labels = []
    test_probabilities = []

    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            features = extractor(data)
            cent_feature, outputs = classifier(features)

            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()
            test_probabilities.extend(probabilities)

            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_predictions.extend(predicted.tolist())
            test_true_labels.extend(labels.tolist())
            
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            
            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    test_accuracy = test_correct / len(test_predictions)
    test_precision = precision_score(test_true_labels, test_predictions, average='weighted')
    test_recall = recall_score(test_true_labels, test_predictions, average='weighted')
    test_f1_score = f1_score(test_true_labels, test_predictions, average='weighted')
    # test_auc = roc_auc_score(test_true_labels, np.exp(test_probabilities), multi_class='ovo')
    test_auc = roc_auc_score(test_true_labels, test_probabilities, multi_class='ovo')

    acc = correct * 100. / total
    err = 100. - acc

    test_confusion_matrix = confusion_matrix(test_true_labels, test_predictions)

    test_specificity = []
    test_sensitivity = []


    for class_label in range(5):
            tp = test_confusion_matrix[class_label, class_label]
            fp = np.sum(test_confusion_matrix[:, class_label]) - tp
            fn = np.sum(test_confusion_matrix[class_label, :]) - tp
            tn = np.sum(test_confusion_matrix) - tp - fp - fn

            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)

            test_specificity.append(specificity)
            test_sensitivity.append(sensitivity)

    results.append({
            'Type': type,
            'Epoch': epoch,
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1_score,
            'AUC': test_auc,
            'Specificity': test_specificity,
            'Sensitivity': test_sensitivity,
            'Confusion Matrix': test_confusion_matrix,
        })

    df = pd.DataFrame(results)
    # Save confusion matrix plot
    os.makedirs(f'./confusion_matrix_{type}/', exist_ok=True)
    plt.figure()
    sns.heatmap(test_confusion_matrix, annot=True, cmap="Blues", fmt="d")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"{type} - Epoch {epoch + 1} Confusion Matrix, ACC:{test_accuracy}")
    plt.savefig(f"./confusion_matrix_{type}/{type}_Epoch{epoch + 1}_confusion_matrix.png")
    plt.close()

    # Create the results directory if it doesn't exist
    os.makedirs(f'./results_{type}/', exist_ok=True)
    df.to_csv(f'./results_{type}/result_{type}_Epoch{epoch + 1}.csv', index=False)

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix=f'test_ON_{type}')

    return acc, err

def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()





