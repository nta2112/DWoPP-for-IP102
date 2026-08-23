# **[Positive Pair Distillation Considered Harmful: Continual Meta Metric Learning for Lifelong Object Re-Identification (BMVC 2022)](https://arxiv.org/abs/2210.01600)**

Here we afford the code to reproduce the experimental results on Market-1501 dataset for our paper *"Positive Pair Distillation Considered Harmful: Continual Meta Metric Learning for Lifelong Object Re-Identification".*

[Kai Wang](https://scholar.google.com/citations?user=j14vd0wAAAAJ), [Chenshen Wu](https://scholar.google.com/citations?user=FO7GyVwAAAAJ&hl=en), [Andy Bagdanov](https://scholar.google.com/citations?user=_Fk4YUcAAAAJ&hl=en), [Xialei Liu* (corresponding)](https://mmcheng.net/xliu/), [Shiqi Yang](https://www.shiqiyang.xyz/), Shangling Jui and [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)

## Datasets

### Market-1501
Market-1501 dataset can be directly downloaded from http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html

After extracting the files, you need to have the following files structure:
```
|-- market1501  
        |-- bounding_box_train  
        |-- bounding_box_test  
        |-- gt_bbox  
        |-- gt_query  
        |-- query  
        |-- readme.txt
```

Or if you have already had the dataset locally, you can create a soft link to it by:
```
ln -s /your/path/to/market1501 ./
```

### IP102 (Pest Recognition)
IP102 dataset with 25 pest classes for lifelong learning (7/6/6/6 task split).

Required files:
- `filtered_class.txt` - 25 category IDs (one per line)
- `classes.txt` - mapping from category ID to class name
- `train.json`, `val.json`, `test.json` - COCO format annotations
- `VOC2007/VOC2007/JPEGImages/` - image files

Auto-discovery supports:
- Local path: `D:/Sau_Benh_object/retrieval-img/IP102 dataset`
- Kaggle input: `/kaggle/input/ip102`
- Environment variable: `IP102_DATA_ROOT`

## Requirements
All python packages in my experimental environment is listed in *requirements.txt*

## Reproducing

### Market-1501, DwoPP(Our method)
```
bash CL_DwoPP.sh
```

### Market-1501, DwPP
```
bash CL_DwPP.sh
```

### Market-1501, FT
```
bash CL_FT.sh
```

### Market-1501, Joint training
```
bash joint_train_dmml_market.sh
```

### IP102, DwoPP (Lifelong Retrieval)
```
bash CL_DwoPP_IP102.sh
```

Or run directly:
```
python CL_train_DwoPP.py \
  --dataset ip102 \
  --dataset_root /path/to/ip102 \
  --filtered_class_path /path/to/filtered_class.txt \
  --classes_txt_path /path/to/classes.txt \
  --num_epochs 200 \
  --num_classes 16 \
  --num_support 5 \
  --num_query 1 \
  --img_height 256 \
  --img_width 128 \
  --gpu 0 \
  --cuda \
  --method DwoPP_IP102 \
  --loss_type dmml \
  --remove_positive_pair
```

## Metrics
- **Retrieval**: R@1, R@5, R@10, mAP (macro)
- **Open-world (OOD)**: Recall@1 (Seen), Recall@1 (Unseen), AUROC, FPR@TPR95
- **Lifelong**: Plasticity, Forgetting, Overall

Results are logged to:
- `results.csv` - per-task metrics
- `history.json` - full history

## Kaggle Notebook
See `kaggle_train.ipynb` for running on Kaggle with auto-clone and dataset detection.

## Others
Other datasets cannot be directly downloaded from their websites due to privacy issues. Please contact the dataset authors. If you have any questions, do not hesitate to contact me or post an issue.