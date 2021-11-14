#based on https://deeplizard.com/learn/video/0LhiS6yu2qQ
import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools

class ConfusionMatrix():
    def __init__(self,classes=["0","1","2","3","4","5","6","7","8","9"]):
        self.classes=classes
        self.cmt = torch.zeros(len(classes),len(classes), dtype=torch.int64)
    def add_batch(self,pred,target):
        stacked = torch.stack(
        (
            target,
            pred
        )
        ,dim=1
        )
        for p in stacked:
            tl, pl = p.tolist()
            self.cmt[tl, pl] = self.cmt[tl, pl] + 1

    def plot_confusion_matrix(self,save_path="conf_matrix.png",normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

        if normalize:
            cm = self.cmt.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
            cm = self.cmt

        print(cm)

        fig, ax = plt.subplots()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)        
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return plt


