import itertools
from torch.nn import functional as F
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from evaluate import FusionMatrix

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'font.size': 10})
classNum=3
rename='MobileNet'

def auc1(trueLabel,abiliable,classes=classNum):
    tempTrueLabel=[0]*len(trueLabel)
    tempAbiliable=[0]*len(trueLabel)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        i+=1
        for j in range(len(trueLabel)):
            if trueLabel[j]==i:
                tempTrueLabel[j]=1
            tempAbiliable[j]=abiliable[j][i-1]
        fpr[i-1], tpr[i-1], thresholds = roc_curve(tempTrueLabel, tempAbiliable, pos_label=1)
        roc_auc[i-1]=auc(fpr[i-1], tpr[i-1])
        tempTrueLabel = [0] * len(trueLabel)
        tempAbiliable = [0] * len(trueLabel)
    return fpr,tpr,roc_auc
def plotPictrue(fpr,tpr,roc_auc):
    lw = 2
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue','red','blue','green','black','bisque','burlywood','antiquewhite','tan','navajowhite',
     'goldenrod','gold','khaki','ivory','forestgreen','limegreen',
     'springgreen','lightcyan','teal','royalblue',
     'navy','slateblue','indigo','darkorchid','darkviolet','thistle']
    save_name=['./logs/'+rename+'-Algorithm-to-0-3-class-disease-sizeClass.png']
    lens=1
    for temp_save_name in save_name:
        for i in range(classNum):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,label='ROC curve of class{0} (AUC area = {1:0.2f})'.format(str(i), roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Sensitivity')
        plt.ylabel('Specificity')
        plt.title(rename)
        plt.legend(loc="lower right")
        plt.savefig(temp_save_name, format='png')
        plt.clf()
        lens+=1
    #plt.show()
def matrixPlot(imagesModelRes,trueLabel):
    cm=confusion_matrix(trueLabel, imagesModelRes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(rename)
    plt.colorbar()
    labels_name=[ str(i) for i in range(classNum)]
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./logs/confusion_matrix.png', format='png')
    # plt.show()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    decoy = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > decoy else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./logs/confusion_matrixV2.png', format='png')


def plot_confusion_matrixV2(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # y_pred = y_pred.argmax(axis=1)
    # y_true = y_true.argmax(axis=1)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
def get_roc_auc(all_preds, all_labels):
    one_hot = label_to_one_hot(all_labels, all_preds.shape[1])

    fpr = {}
    tpr = {}
    roc_auc = np.zeros([all_preds.shape[1]])
    for i in range(all_preds.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(one_hot[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc


def label_to_one_hot(label, num_class):
    one_hot = F.one_hot(torch.from_numpy(label).long(), num_class).float()
    one_hot = one_hot.numpy()

    return one_hot


def index_calculation(all_preds, all_result, all_labels, log, num_classes=5):
    fusion_matrix = FusionMatrix(num_classes)
    fusion_matrix.update(all_result, all_labels)
    roc_auc = get_roc_auc(all_preds, all_labels)

    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["specificity"] = fusion_matrix.get_pre_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = roc_auc
    metrics["fusion_matrix"] = fusion_matrix.matrix

    metrics["acc"] = fusion_matrix.get_accuracy()
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    auc_mean = np.mean(metrics["roc_auc"])
    spec_mean = np.mean(metrics["specificity"])

    print("\n-------  Valid result: Valid_Acc: {:>6.3f}%  Balance_Acc: {:>6.3f}%  -------".format(
        metrics["acc"] * 100, metrics["bacc"] * 100))
    log.write(("\n-------  Valid result: Valid_Acc: {:>6.3f}%  Balance_Acc: {:>6.3f}%  -------\n".format(
        metrics["acc"] * 100, metrics["bacc"] * 100)))

    print("         roc_auc.mean: {:>6.3f}  f1_score: {:>6.4f}     ".format(
        auc_mean, metrics["f1_score"]))
    log.write("roc_auc.mean: {:>6.3f}  f1_score: {:>6.4f}\n".format(
        auc_mean, metrics["f1_score"]))

    print("         roc_auc:       {}  ".format(metrics["roc_auc"]))
    log.write("roc_auc:       {}\n".format(metrics["roc_auc"]))

    print("         sensitivity:   {}  ".format(metrics["sensitivity"]))
    log.write("sensitivity:   {}\n".format(metrics["sensitivity"]))

    print("         specificity:   {}   mean:   {}  ".format(metrics["specificity"], spec_mean))
    log.write("specificity:   {}   mean:   {}\n".format(metrics["specificity"], spec_mean))

    print("         fusion_matrix: \n{}  ".format(metrics["fusion_matrix"]))
    log.write("fusion_matrix: \n{}\n".format(metrics["fusion_matrix"]))
