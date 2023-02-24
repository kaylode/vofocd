import json
from pathlib import Path
import cv2
from natsort import natsorted
import numpy as np
from tqdm import tqdm
from pprint import pprint
import subprocess

BGRforLabel = { 
               0: (205, 117, 149),
               1: (142, 207, 159), 
               2: (0, 255, 255),   
               3: (225, 208, 77),
               4: (107, 121, 0),
               5: (0, 0, 255),
            }

class Infer:
    
    def __init__(
        self,
        img_root,
        gt_json_path,
        pred_json_path,
        gt_txt_dir,
        pred_txt_dir,
        gt_img_size=(480, 360), # w,h
        pred_img_size=(480, 480), # w,h
        pred_format="xywh",
        num_classes=6
    ):
        self.img_root= img_root
        self.gt_json_path = gt_json_path
        self.pred_json_path = pred_json_path
        self.gt_img_size = gt_img_size
        self.pred_img_size = pred_img_size
        self.pred_format = pred_format
        self.num_classes = num_classes
        self.gt_data, self.pred_data = self.get_gt_pred_data()
        self.wrong_instances = self.get_wrong_instances()
        self.gt_txt_dir = gt_txt_dir
        self.pred_txt_dir = pred_txt_dir
    
    def get_gt_pred_data(self):
    
        _gt_data = {}
        _pred_data = {}

        with open(self.gt_json_path) as json_file:
            gt_data = json.load(json_file)
        
        for image in gt_data['images']:
            image_id, image_name = image['id'], Path(image['file_name']).name.split("_")[-1]
            id = image_name
        
            for annotations in gt_data['annotations']:

                if annotations['image_id'] == image_id:

                    if id not in _gt_data.keys():
                        _gt_data[id] = {
                            "category_id": [],
                            "bbox": []
                        }
                    
                    if annotations["category_id"] - 1 not in [2, 3]:
                        continue
                    
                    # _gt_data[id]['category_id'].append(annotations['category_id'] - 1)
                    _gt_data[id]['category_id'].append(0)
                    _gt_data[id]['bbox'].append(annotations['bbox'])
        
        with open(self.pred_json_path) as json_file:
            pred_data = json.load(json_file)

        gt_x, gt_y = [float(x) for x in self.gt_img_size]
        pred_x, pred_y = [float(x) for x in self.pred_img_size]
        
        for pred in pred_data:
            id = pred["image_name"]
            if id in _gt_data.keys():
                _pred_data[id] = {k: v for k, v in pred.items() if k != "image_name"}
                _pred_data[id]['category_id'] = np.array(_pred_data[id]['category_id'])
                _pred_data[id]["bbox"] = np.array(_pred_data[id]["bbox"])

                # Normalized and resize to gt_img_size
                if len(_pred_data[id]["bbox"] >= 2):
                    _pred_data[id]["bbox"][:, 0] = _pred_data[id]["bbox"][:, 0] / pred_x * gt_x
                    _pred_data[id]["bbox"][:, 1] = _pred_data[id]["bbox"][:, 1] / pred_y * gt_y
                    _pred_data[id]["bbox"][:, 2] = _pred_data[id]["bbox"][:, 2] / pred_x * gt_x
                    _pred_data[id]["bbox"][:, 3] = _pred_data[id]["bbox"][:, 3] / pred_y * gt_y
                
                    # xyxy to xywh
                    if self.pred_format == "xyxy":
                        _pred_data[id]["bbox"][:, 3] = _pred_data[id]["bbox"][:, 3] - _pred_data[id]["bbox"][:, 1]
                        _pred_data[id]["bbox"][:, 2] = _pred_data[id]["bbox"][:, 2] - _pred_data[id]["bbox"][:, 0]
                    _pred_data[id]["bbox"] = _pred_data[id]["bbox"].tolist()

        _gt_data = {k:v for k, v in natsorted(_gt_data.items())}
        _pred_data = {k:v for k, v in natsorted(_pred_data.items())}
        return _gt_data, _pred_data 

    def seek(self):
        for ix, (k, v) in enumerate(self.gt_data.items()):
            print(k, v)
            break

        for ix, (k, v) in enumerate(self.pred_data.items()):
            print(k, v)
            break
    
    def generate_gt_txt(self, out_format="xywh"):
        Path(self.gt_txt_dir).mkdir(exist_ok=True)
        for k, v in self.gt_data.items():
            with open(Path(self.gt_txt_dir) / "{}.txt".format(k.split('.')[0]), 'w') as f:
                for cls, bbox in zip(v['category_id'], v['bbox']):
                    x, y, w, h = bbox
                    if out_format == "xywh":
                        to_write = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cls, x, y, w, h)
                    f.write(to_write)

    def generate_pred_txt(self, out_format="xywh", min_conf=None):
        Path(self.pred_txt_dir).mkdir(exist_ok=True)
        for k, v in self.pred_data.items():
            with open(Path(self.pred_txt_dir) / "{}.txt".format(k.split('.')[0]), 'w') as f:
                for score, cls, bbox in zip(v['score'], v['category_id'], v['bbox']):
                    if min_conf and score < min_conf:
                        continue
                    x, y, w, h = bbox
                    if out_format == "xywh":
                        to_write = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cls, score, x, y, w, h)
                    # elif out_format == "cxcywh":
                    #     w = x2 - x1
                    #     h = y2 - y1
                    #     cx = (x1 + w / 2.0) / 480.0
                    #     cy = (y1 + h / 2.0) / 360.0
                    #     w = w / 480.0
                    #     h = h / 360.0
                    #     to_write = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cls, score, cx, cy, w, h)
                    f.write(to_write)
    
    def __get_stat_dict(self):
        return {k: v for k, v in zip(range(self.num_classes), [0]*self.num_classes)}
                
    def get_stat_dict(self, data):
        stat_dict = self.__get_stat_dict()
        for k, v in data.items():
            for cls in v["category_id"]:
                stat_dict[cls] += 1
        return stat_dict
       
    def get_wrong_instances(self):
        
        self.wrong_instances = []
        count = 0
        
        for k, v in self.pred_data.items():
            if not np.array_equal(
                natsorted(np.array(v["category_id"], dtype=np.int32)), 
                natsorted(np.array(self.gt_data[k]["category_id"], dtype=np.int32))
            ):
                self.wrong_instances.append(k)
                count += 1
        print("Percent of wrong instances:", count / len(self.gt_data) * 100)
        return self.wrong_instances

    def get_missing_instances_stats(self):
        
        stat_dict = self.__get_stat_dict()
        
        for ins in self.wrong_instances:
            gt_cls, pred_cls = self.gt_data[ins]["category_id"], self.pred_data[ins]["category_id"]
            set_gt_cls, set_pred_cls = natsorted(list(set(gt_cls))), natsorted(list(set(pred_cls)))
            for cls in set_gt_cls:
                if cls not in set_pred_cls:
                    stat_dict[cls] += 1
        
        for k, v in stat_dict.items():
            stat_dict[k] = np.round(v / len(self.wrong_instances) * 100.0, 2)
        return stat_dict

    def get_redundant_instances_stats(self):
        
        stat_dict = self.__get_stat_dict()
        
        for ins in self.wrong_instances:
            gt_cls, pred_cls = self.gt_data[ins]["category_id"], self.pred_data[ins]["category_id"]
            set_gt_cls, set_pred_cls = natsorted(list(set(gt_cls))), natsorted(list(set(pred_cls)))
            unique, counts = np.unique(gt_cls, return_counts=True)
            gt_cls_count = dict(zip(unique, counts))
            unique, counts = np.unique(pred_cls, return_counts=True)
            pred_cls_count = dict(zip(unique, counts))
            for cls in set_gt_cls:
                if cls in set_pred_cls and pred_cls_count[cls] > gt_cls_count[cls]:
                    stat_dict[cls] += 1
        
        for k, v in stat_dict.items():
            stat_dict[k] = np.round(v / len(self.wrong_instances) * 100.0, 2)
        return stat_dict

    def summary(self):
        print("gt:   ", self.get_stat_dict(self.gt_data))
        print("bbox class stats")
        print("pred: ", self.get_stat_dict(self.pred_data))
        print("missing bbox class stats")
        print("pred: ", self.get_missing_instances_stats())
        print("redundant bbox class stats")
        print("pred: ", self.get_redundant_instances_stats())
        
    def get_mAP(self, save_path):
        Path(save_path).mkdir(exist_ok=True)
        subprocess.run([
            "bash", 
            "/home/htluc/Object-Detection-Metrics/slurm_run.sh", 
            self.gt_txt_dir,
            self.pred_txt_dir,
            save_path,
        ])

    def visualize_test(self, img_name="20210222072133.jpg"):
        img_root = Path(self.img_root)
        for k, v in self.gt_data.items():
            if k == img_name:
                self.visualize_per_image(
                    img_path=str(img_root / k), 
                    cls=v['category_id'], 
                    bbox=v['bbox'],
                    save_img=True
                )
                break
        
        for k, v in self.pred_data.items():
            if k == img_name:
                self.visualize_per_image(
                    img_path=str(img_root / k), 
                    cls=v['category_id'], 
                    bbox=v['bbox'],
                    score=v['score'],
                    save_img=True
                )
                break

    def visualize_per_image(self, img_path, save_path, cls, bbox, score=None, save_img=True):
        img = cv2.imread(img_path)
        if score:
            for _score, _cls, _bbox in zip(score, cls, bbox):
                if score < 0.25:
                    continue
                x, y, w, h  = np.array(_bbox, dtype=np.int32)
                x1 = x
                y1 = y
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(img, (x1, y1), (x2, y2), BGRforLabel[_cls], 2)
                cv2.putText(img, "{}:{:.1f}%".format(_cls, _score*100), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGRforLabel[_cls], 2)
            if save_img:
                cv2.imwrite(str(Path(save_path) / "{}_det.png".format(Path(img_path).stem)), img)
        else:
            for _cls, _bbox in zip(cls, bbox):
                x, y, w, h  = np.array(_bbox, dtype=np.int32)
                x1 = x
                y1 = y
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(img, (x1, y1), (x2, y2), BGRforLabel[_cls], 2)
                cv2.putText(img, str(_cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGRforLabel[_cls], 1)
            if save_img:
                cv2.imwrite(str(Path(save_path) / "{}_gt.png".format(Path(img_path).stem)), img)
        return img
    
    def _visualize_side_by_side(self, gt, pred, img_path, save_path):
        gt_img = self.visualize_per_image(img_path, save_path, gt['category_id'], gt['bbox'], save_img=False)
        pred_img = self.visualize_per_image(img_path, save_path, pred['category_id'], pred['bbox'], pred['score'], save_img=False)
        sbs_img = np.hstack((gt_img, pred_img))
        dest = Path(save_path) / "{}.png".format(Path(img_path).stem)
        cv2.imwrite(str(dest), sbs_img)
        
    def visualize_side_by_side(self, save_path):
        Path(save_path).mkdir(exist_ok=True)
        pbar = tqdm(enumerate(zip(self.gt_data.items(), self.pred_data.items())))
        for ix, (x, y) in pbar:
            gt_k, gt_v = x
            pred_k, pred_v = y
            pbar.set_description_str(gt_k)
            if not gt_k == pred_k:
                print("something wrong in index", ix)
            self._visualize_side_by_side(
                gt_v, pred_v, str(Path(self.img_root) / gt_k), 
                save_path=save_path
            )
 
def main():
    # infer = Infer(
    #     img_root="/home/htluc/datasets/aim_folds/fold_0/val/images",
    #     gt_json_path="/home/htluc/datasets/aim/annotations/annotation_0_val.json",
    #     pred_json_path="/home/htluc/vocal-folds/runs/detr_resnet18_0_infer/json/result.json",
    #     gt_txt_dir="/home/htluc/datasets/aim_folds/infer/detr_resnet18_0/gt",
    #     pred_txt_dir="/home/htluc/datasets/aim_folds/infer/detr_resnet18_0/pred",
    #     gt_img_size=(480, 360),
    #     pred_img_size=(480, 480),
    #     pred_format="xyxy"
    # )
    # infer.seek()
    # infer.visualize_side_by_side("/home/htluc/datasets/aim_folds/infer/detr_resnet18_0/sbs")
    # infer.summary()
    # # !!! get_mAP will prompt to delete the save_path_folder
    # infer.get_mAP(save_path="/home/htluc/datasets/aim_folds/infer/detr_resnet18_0/mAP_result")
    
    folder = "lesions_only"
    
    infer = Infer(
        img_root="/home/htluc/datasets/aim_folds/fold_0/val/images",
        gt_json_path="/home/htluc/datasets/aim/annotations/annotation_0_val.json",
        pred_json_path="/home/htluc/yolov5/runs/val/yolov5s_lesions_fold_0_val/better.json",
        gt_txt_dir="/home/htluc/datasets/aim_folds/infer/yolov5s_lesions/{}/gt".format(folder),
        pred_txt_dir="/home/htluc/datasets/aim_folds/infer/yolov5s_lesions/{}/pred".format(folder),
        gt_img_size=(480, 360),
        pred_img_size=(480, 360),
        pred_format="xywh"
    )
    infer.seek()
    infer.generate_gt_txt()
    infer.generate_pred_txt()
    # infer.visualize_side_by_side("/home/htluc/datasets/aim_folds/infer/yolov5s_lesions/{}/sbs".format(folder))
    infer.summary()
    # !!! get_mAP will prompt to delete the save_path_folder
    infer.get_mAP(save_path="/home/htluc/datasets/aim_folds/infer/yolov5s_lesions/{}/mAP_result".format(folder))
    
if __name__ == "__main__":
    main()