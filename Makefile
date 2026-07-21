train:
	python train_model.py \
	--model ConvNeXtB \
	--num_classes 9 \
	--embed_dim 1024 \
	--epochs 50 \
	--optimizer adamw \
	--lr 0.0001 \
	--weight_decay 0.00 \
	--db_dir "/home/ahanaf/Datasets/" \
	--train_db Eye_Disease_Image_Dataset \
	--test_db Eye_Disease_Image_Dataset \
	--working_dir "/home/ahanaf/Desktop/ScRp2026/Experiments" \
	--image_size 224 \
	--batch_size 32 \
	--db_worker_count 16 \
	--seed 42 \
	--grad_clip_norm 1.0 \
	--loss_fn crossentropy \
	--use_ema True \
	--ema_decay 0.999 \
	--early_stopping_patience 15 \
	--sampling_mode under_to_N \
	--sample_size 500 \

test:
	python test_model.py \
		--snapshot "/home/ahanaf/Desktop/ScRp2026/Experiments/experiment-0004" \
		--db_dir "/home/ahanaf/Datasets/" \
		--test_db Eye_Disease_Image_Dataset \


ensemble:
	python test_ensemble.py \
	--snapshot_dir "/home/ahanaf/ScRp2026/Experiments/" \
	--snapshots 8,15,17 \
	--ensemble_mode max \
	--db_dir "/home/ahanaf/Datasets/" \
	--test_db Eye_Disease_Image_Dataset \


bootstrap:
	python bootstrap_ci.py \
	--csv /home/ahanaf/ScRp2026/Experiments/experiment-0023/predictions.csv \
	--n_boot 1000 \
	--ci 95 \
	--classmap /home/ahanaf/Datasets/Eye_Disease_Image_Dataset/class_map.json \
	--seed 0 \


tensorboard:
	tensorboard --logdir /home/ahanaf/Desktop/EyeDisease2026/Experiments/experiment-0000


clean:
	rm -rf __pycache__
	rm -rf */__pycache__



