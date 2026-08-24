import os
import json
import pickle as pkl
from collections import defaultdict
import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import numpy as np
import random

from data.common import list_pictures


class IP102(Data.Dataset):
    def __init__(self, dataset_root, transform, task_id=0, split='train',
                 ROOT_PATH='preprocess_dataset/', dataset_name='ip102',
                 filtered_class_path=None, classes_txt_path=None):
        if split not in ['train', 'gallery', 'query', 'val', 'test']:
            raise Exception('Invalid dataset split.')
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name

        self.filtered_classes = self._load_filtered_classes(filtered_class_path)
        self.class_names = self._load_class_names(classes_txt_path)
        self.task_splits = self._create_task_splits()

        if split == 'train':
            file_name = f'ip102_task_splits.pkl'
            preprocess_path = os.path.join(ROOT_PATH, dataset_name, file_name)
            
            if os.path.exists(preprocess_path):
                with open(preprocess_path, 'rb') as f:
                    CL_data_split = pkl.load(f)
            else:
                CL_data_split = self._preprocess_and_save(preprocess_path)

            curr_task_split = CL_data_split[task_id]
            self.train_label = curr_task_split['train_label']
            self.train_data = curr_task_split['train_data']
            print(f'Load task {task_id} train data: {len(self.train_label)} samples, {len(set(self.train_label))} classes')

        else:
            # Map gallery/query to val/test as appropriate
            if split in ['gallery', 'query']:
                json_split = 'test'
            else:
                json_split = split
            
            json_file = os.path.join(dataset_root, f'{json_split}.json')
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
            
            self.coco_data = coco_data
            self.img_info = {img['id']: img for img in coco_data['images']}
            self.anns_by_img = defaultdict(list)
            for ann in coco_data['annotations']:
                if ann['category_id'] in self.filtered_classes:
                    self.anns_by_img[ann['image_id']].append(ann)
            
            self.img_ids = [img_id for img_id, anns in self.anns_by_img.items() if len(anns) > 0]
            print(f'Load {split} data: {len(self.img_ids)} images')

    def _load_filtered_classes(self, path):
        if path is None:
            possible_paths = [
                'D:/Sau_Benh_object/retrieval-img/IP102 dataset/filtered_class.txt',
                '/kaggle/input/ip102/filtered_class.txt',
                'filtered_class.txt',
                '../IP102 dataset/filtered_class.txt',
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    path = p
                    break
        
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                classes = [int(line.strip()) for line in f if line.strip()]
            print(f'Loaded {len(classes)} filtered classes from {path}')
            return classes
        else:
            raise FileNotFoundError('filtered_class.txt not found. Please provide path.')

    def _load_class_names(self, path):
        if path is None:
            possible_paths = [
                'D:/Sau_Benh_object/retrieval-img/IP102 dataset/classes.txt',
                '/kaggle/input/ip102/classes.txt',
                'classes.txt',
                '../IP102 dataset/classes.txt',
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    path = p
                    break
        
        class_names = {}
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        class_names[int(parts[0])] = parts[1].strip()
            print(f'Loaded {len(class_names)} class names from {path}')
        return class_names

    def _create_task_splits(self):
        task_sizes = [7, 6, 6, 6]
        task_splits = []
        start = 0
        for size in task_sizes:
            task_splits.append(self.filtered_classes[start:start+size])
            start += size
        return task_splits

    def _preprocess_and_save(self, save_path):
        json_file = os.path.join(self.dataset_root, 'train.json')
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        img_info = {img['id']: img for img in coco_data['images']}
        anns_by_img = defaultdict(list)
        for ann in coco_data['annotations']:
            if ann['category_id'] in self.filtered_classes:
                anns_by_img[ann['image_id']].append(ann)
        
        class_to_imgs = defaultdict(list)
        for img_id, anns in anns_by_img.items():
            for ann in anns:
                class_to_imgs[ann['category_id']].append(img_id)
        
        CL_data_split = []
        for task_id, task_classes in enumerate(self.task_splits):
            train_data = []
            train_label = []
            for cls in task_classes:
                for img_id in class_to_imgs.get(cls, []):
                    img = img_info[img_id]
                    file_name = img['file_name']
                    img_path = os.path.join(self.dataset_root, file_name)
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.dataset_root, 'VOC2007', 'VOC2007', 'JPEGImages', file_name)
                    if os.path.exists(img_path):
                        train_data.append(file_name)
                        train_label.append(cls)
            
            CL_data_split.append({
                'train_data': train_data,
                'train_label': train_label,
                'classes': task_classes
            })
            print(f'Task {task_id}: {len(task_classes)} classes, {len(train_data)} images')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pkl.dump(CL_data_split, f)
        print(f'Saved preprocessed splits to {save_path}')
        
        return CL_data_split

    def __getitem__(self, index):
        if self.split == 'train':
            ref_path = self.train_data[index]
            path = os.path.join(self.dataset_root, ref_path)
            if not os.path.exists(path):
                path = os.path.join(self.dataset_root, 'VOC2007', 'VOC2007', 'JPEGImages', ref_path)
            label = self.train_label[index]
        else:
            img_id = self.img_ids[index]
            img_info = self.img_info[img_id]
            file_name = img_info['file_name']
            path = os.path.join(self.dataset_root, file_name)
            if not os.path.exists(path):
                path = os.path.join(self.dataset_root, 'VOC2007', 'VOC2007', 'JPEGImages', file_name)
            
            anns = self.anns_by_img[img_id]
            label = anns[0]['category_id']

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if self.split == 'train':
            return len(self.train_label)
        else:
            return len(self.img_ids)

    @property
    def unique_ids(self):
        if self.split == 'train':
            return sorted(set(self.train_label))
        else:
            labels = [self[i][1] for i in range(len(self))]
            return sorted(set(labels))

    def get_class_name(self, class_id):
        return self.class_names.get(class_id, str(class_id))

    @property
    def task_classes(self):
        return self.task_splits