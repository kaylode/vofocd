import torch
import numpy as np
from sklearn.metrics import f1_score

def postprocess_multilabel(tensor, threshold=0.5):
    tensor = torch.sigmoid(torch.from_numpy(tensor))
    tensor = tensor.numpy()
    tensor = tensor > threshold
    outputs = tensor.astype(np.int)
    return outputs

def postprocess_multiclass(tensor):
    tensor = torch.argmax(torch.from_numpy(tensor),dim=1)
    tensor = tensor.numpy()
    outputs = tensor.astype(np.int)
    return outputs

def get_postprocess_fn(types):
    if types == 'multiclass':
        return postprocess_multiclass
    if types == 'multilabel':
        return postprocess_multilabel

class F1ScoreMetric():
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, types="multiclass", average = 'weighted'):
        self.average = average
        self.types = types
        self.postprocess_fn = get_postprocess_fn(types)
        self.reset()

    def update(self, outputs, targets):
        pred = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        self.targets.append(targets)
        self.preds.append(pred)

    def compute(self): 
        self.targets = np.concatenate(self.targets)
        self.preds = np.concatenate(self.preds)
        
        self.preds = self.postprocess_fn(self.preds)
        return f1_score(self.targets, self.preds, average=self.average)

    def value(self):
        score = self.compute()
        return {"f1-score" : score}

    def reset(self):
        self.targets = []
        self.preds = []
    
    def __str__(self):
        return f'F1-Score: {self.value()}'

if __name__ == '__main__':
    f1 = F1ScoreMetric()
    y_true = torch.tensor([[1,0,1],[1,1,0],[0,0,0]])
    y_pred = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    f1.update(y_pred ,y_true)
    f1.update(y_pred ,y_true)
    print(f1.value())

