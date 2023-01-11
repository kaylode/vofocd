from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
import numpy as np
import argparse
import json
import os.path as osp
from theseus.base.utilities.loggers import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-i',
                        type=str,
                        help='path to json file')
    parser.add_argument('--num_folds', '-k',
                        type=int,
                        default=5,
                        help='number of folds (default: 5)')
    parser.add_argument('--seed',
                        type=int,
                        default=1702,
                        help='random seed (default: 1702)')
    return parser.parse_args()


def save_to_coco(target_image_ids, images, categories, annotations, save_path=None):

    json_dict = {
        'categories': categories,
        'annotations': [],
        'images': []
    }

    for item in images:
        image_id = item['id']
        if image_id in target_image_ids:
            json_dict['images'].append(item)

    for item in annotations:
        image_id = item['image_id']
        if image_id in target_image_ids:
            json_dict['annotations'].append(item)     

    json.dump(json_dict, open(save_path, 'w'))
        
def split_coco(args):
    # Read CSV
    coco = json.load(open(args.json))

    # Set random seed
    np.random.seed(args.seed)

    images = coco['images'] 
    annotations = coco['annotations']
    categories = coco['categories']
    number_of_images = len(images)
    
    image_class_dict = {}
    for item in annotations:
        image_id = item['image_id']
        class_id = item['category_id']
        if image_id not in image_class_dict.keys():
            image_class_dict[image_id] = []
        image_class_dict[image_id].append(class_id)

    data = [[k, v] for k, v in image_class_dict.items()]
    
    # Transform into numpy arrays
    X, y = zip(*data)
    X = np.array(X)

    # Get unique classes
    original_tokens = sum(y, [])

    # # Transform into binary vectors (1=class appears)
    unique_tokens = sorted(list(set(original_tokens)))
    mlb = MultiLabelBinarizer(classes=unique_tokens)
    y_bin = mlb.fit_transform(y)


    # k-fold split
    X_indices = np.array(range(len(X))).reshape(-1, 1)
    k_fold = IterativeStratification(n_splits=args.num_folds, order=1)
    k_fold_split = k_fold.split(X_indices, y_bin)

    # Assign fold to each id
    id2fold = dict()
    for i, (_, val_indices) in enumerate(k_fold_split):
        for x in X[val_indices]:
            id2fold[x] = i

    basename, _ = osp.splitext(osp.basename(args.json))
    savedir = osp.dirname(args.json)
    for fold in range(args.num_folds):
        val_ids = [i for i,f in id2fold.items() if f==fold]
        train_ids = [i['id'] for i in images if i['id'] not in val_ids]

        save_to_coco(val_ids, images, categories, annotations, save_path=osp.join(savedir, basename+f'_{fold}_val.json'))
        save_to_coco(train_ids, images, categories, annotations, save_path=osp.join(savedir, basename+f'_{fold}_train.json'))

    LOGGER.text(f'Save folds to {savedir}', level=LoggerObserver.INFO)

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    split_coco(args)