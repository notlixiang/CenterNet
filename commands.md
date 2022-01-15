# coco test
python test.py ctdet --exp_id coco_dla --keep_res --load_model ../models/ctdet_coco_dla_2x.pth --batch_size 4

# coco train
python main.py ctdet --exp_id coco_dla --batch_size 4 --lr 1.25e-4  --gpus 0
python src/main.py ctdet --exp_id coco_resdcn18 --arch res_18 --batch_size 2 --lr 5e-4 --gpus 0 --amp