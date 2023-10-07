python3 ./train.py --arch Uformer_B --batch_size 16 --gpu '0,1,2,3' \
    --train_ps 128 --train_dir ../datasets/denoising/train --env _0706 \
    --val_dir ../datasets/denoising/val --save_dir ./logs/ \
    --dataset mydataset --warmup 