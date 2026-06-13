import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
EarthMiss_CATEGORIES = [
    {"color": [255, 255, 255], "id": 1, "name": "Background"},
    {"color": [255, 0, 0], "id": 2, "name": "Building"},
    {"color": [255, 255, 0], "id": 3, "name": "Road"},
    {"color": [0, 0, 255], "id": 4, "name": "Water"},
    {"color": [159, 129, 183], "id": 5, "name": "Barren"},
    {"color": [0, 255, 0], "id": 6, "name": "Forest"},
    {"color": [255, 195, 128], "id": 7, "name": "Agricultural"},
    {"color": [165, 0, 165], "id": 8, "name": "Playground"},
]

def _get_EarthMiss_meta():
    stuff_ids = [k["id"] for k in EarthMiss_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in EarthMiss_CATEGORIES]
    stuff_colors = [k["color"] for k in EarthMiss_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_EarthMiss(root):
    meta = _get_EarthMiss_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "EarthMiss/images", "EarthMiss/labels"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"EarthMiss_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "/data1/ruizhong_data/GHRLandCover"
register_EarthMiss(root)
