import torch
import numpy as np

def compute_multiclass(output, target):
    pred = torch.argmax(output, dim=1)
    correct = (pred == target).sum()
    sample_size = output.size(0)
    return correct, sample_size

def compute_multilabel(output, target, thresh=0.5):
    output = torch.sigmoid(output)
    preds = output > thresh
    preds = preds.long()
    correct = (preds == target)
    results = 0
    for i in correct:
        results += i.all()
    sample_size = target.shape[0]
    return results, sample_size

def compute_binary(output, target, thresh=0.7):
    pred = output >= thresh
    correct = (pred == target).sum()
    sample_size = output.size(0)
    return correct, sample_size

def get_compute(types):
    if types == 'binary':
        return compute_binary
    elif types == 'multiclass':
        return compute_multiclass
    elif types == 'multilabel':
        return compute_multilabel

class AccuracyMetric():
    """
    Accuracy metric for classification
    """
    def __init__(self, types = 'multiclass', decimals = 10):
        self.reset()
        self.decimals = decimals
        self.compute_fn = get_compute(types)
        
    def compute(self, outputs, targets):
        return self.compute_fn(outputs, targets)

    def update(self,  outputs, targets):
        assert isinstance(outputs, torch.Tensor), "Please input tensors"
        outputs = outputs.squeeze().cpu()
        targets = targets.cpu()
        results, sample_size = self.compute(outputs, targets)
        self.correct += results.numpy()
        self.sample_size += sample_size

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        values = self.correct / self.sample_size
        return {"acc" : np.around(values, decimals = self.decimals)}

    def __str__(self):
        return f'Accuracy: {self.value()}'

    def __len__(self):
        return len(self.sample_size)

if __name__ == '__main__':
    accuracy = AccuracyMetric(decimals = 4, types='multilabel')
    out = [[1,0,1],[1,1,0],[0,0,0]]
    label = [[1, 1, 0], [1, 1, 0], [1, 1, 0]]
    outputs = torch.LongTensor(out)
    targets = torch.LongTensor(label)
    accuracy.update(outputs, targets)
    di = {}
    di.update(accuracy.value())
    print(di)
    
  
