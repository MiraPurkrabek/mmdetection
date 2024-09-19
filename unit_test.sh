# # Bbox detection
# python demo/image_demo.py \
#     ../data/OCHuman/COCO-like/val2017/000001.jpg \
#     configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py \
#     --weights models/pretrained/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth

# Instance segmentation
python demo/image_demo.py \
    ../data/OCHuman/COCO-like/val2017/000001.jpg \
    configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py \
    --weights models/pretrained/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth