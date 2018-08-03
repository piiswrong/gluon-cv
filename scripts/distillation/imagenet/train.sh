# !/bin/bash
echo $@

python distill_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model resnet50_v1b --mode hybrid \
  --lr 0.2 --lr-mode cosine \
  --num-epochs 200 --batch-size 64 --num-gpus 8 -j 60 \
  --warmup-epochs 5 --use-rec --last-gamma \
  --hard-weight=0.5 --temperature=20 --mixup=0.2 \
  --teacher=resnet101_v1c \
  $@
