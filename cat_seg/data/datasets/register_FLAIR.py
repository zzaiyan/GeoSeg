import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# 根据图片生成的 FLAIR 类别配置 (12类)
FLAIR_CATEGORIES = [
    {"color": [200, 50, 140], "id": 1, "name": "building"},
    {"color": [140, 135, 125], "id": 2, "name": "pervious surface"},
    {"color": [235, 55, 35], "id": 3, "name": "impervious surface"},
    {"color": [165, 115, 40], "id": 4, "name": "bare soil"},
    {"color": [45, 80, 165], "id": 5, "name": "water"},
    {"color": [40, 70, 50], "id": 6, "name": "coniferous"},
    {"color": [120, 220, 135], "id": 7, "name": "deciduous"},
    {"color": [230, 165, 55], "id": 8, "name": "brushwood"},
    {"color": [85, 20, 120], "id": 9, "name": "vineyard"},
    {"color": [155, 245, 80], "id": 10, "name": "herbaceous vegetation"},
    {"color": [255, 235, 85], "id": 11, "name": "agricultural land"},
    {"color": [230, 225, 155], "id": 12, "name": "plowed land"},
    # {"color": [0, 0, 0], "id": 13, "name": "other"},
]

def _get_FLAIR_meta():
    stuff_ids = [k["id"] for k in FLAIR_CATEGORIES]
    # 将数据集 ID (1-12) 映射到训练连续 ID (0-11)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in FLAIR_CATEGORIES]
    stuff_colors = [k["color"] for k in FLAIR_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_FLAIR(root):
    meta = _get_FLAIR_meta()
    
    # -------------------------------------------------------
    # ⚠️ 请根据实际文件夹结构修改下面的路径名称
    # 假设你的数据集文件夹名为 "FLAIR" 且内部结构与之前类似
    # -------------------------------------------------------
    for name, image_dirname, sem_seg_dirname in [
        ("train", "FLAIR_split/train/images", "FLAIR_split/train/labels"),
        ("val", "FLAIR_split/val/images", "FLAIR_split/val/labels"),
        ("all", "FLAIR/images", "FLAIR/labels"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        dataset_name = f"FLAIR_{name}_sem_seg"
        
        # 注册数据集
        # 注意: gt_ext 和 image_ext 根据你的实际文件后缀修改 (如 jpg, png, tif)
        DatasetCatalog.register(
            dataset_name, 
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        
        # 设置元数据
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


root = "/data1/ruizhong_data/GHRLandCover"
register_FLAIR(root)