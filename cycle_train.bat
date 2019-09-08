@echo off

TITLE %TITLE%
:LOOP

python train.py --logtostderr --train_dir=training/  --pipeline_config_path=training/ssd_mobilenet_v1_coco.config

GoTo:LOOP