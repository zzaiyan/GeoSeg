import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# 根据 OEM 图片生成的类别配置 (ID 1-8 映射到 0-7)
OEM_CATEGORIES = [
    {"color": [128, 0, 0], "id": 1, "name": "Bareland"},
    {"color": [0, 255, 36], "id": 2, "name": "Rangeland"},
    {"color": [148, 148, 148], "id": 3, "name": "Developed space"},
    {"color": [255, 255, 255], "id": 4, "name": "Road"},
    {"color": [34, 97, 38], "id": 5, "name": "Tree"},
    {"color": [0, 69, 255], "id": 6, "name": "Water"},
    {"color": [75, 181, 73], "id": 7, "name": "Agriculture land"},
    {"color": [222, 31, 7], "id": 8, "name": "Building"},
]

def _get_OEM_meta():
    stuff_ids = [k["id"] for k in OEM_CATEGORIES]
    # 建立映射：{1:0, 2:1, ..., 8:7}
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in OEM_CATEGORIES]
    stuff_colors = [k["color"] for k in OEM_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_OEM(root):
    meta = _get_OEM_meta()
    # 根据您的实际 OEM 目录结构调整子路径
    for name, image_dirname, sem_seg_dirname in [
        ("train", "OEM_split/train/images", "OEM_split/train/labels"),
        ("val", "OEM_split/val/images", "OEM_split/val/labels"),
        ("all", "OEM/images", "OEM/labels"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        dataset_name = f"OEM_{name}_sem_seg"
        
        DatasetCatalog.register(
            dataset_name, 
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  
            **meta,
        )

# 设置您的数据集根目录
root = "/data1/ruizhong_data/GHRLandCover"
register_OEM(root)