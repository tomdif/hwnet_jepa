.PHONY: help install synthetic-supervised synthetic-jepa synthetic-transfer cifar10-jepa stl10-jepa clean

help:
	@echo "Targets:"
	@echo "  install              - pip install -e ."
	@echo "  synthetic-supervised - run supervised baselines on synthetic data"
	@echo "  synthetic-jepa       - run JEPA pretraining + probe on synthetic data"
	@echo "  synthetic-transfer   - run transfer experiment on synthetic data"
	@echo "  cifar10-jepa         - run JEPA pretraining + probe on CIFAR-10"
	@echo "  stl10-jepa           - run JEPA pretraining + probe on STL-10"
	@echo "  clean                - rm -rf results/ data/"

install:
	pip install -e .

synthetic-supervised:
	python scripts/run_supervised.py --source synthetic_10class --seeds 6

synthetic-jepa:
	python scripts/run_jepa.py --source synthetic_10class \
		--pretrain_epochs 12 --seeds 6 \
		--readouts linear attn_pool

synthetic-transfer: synthetic-jepa
	python scripts/run_transfer.py \
		--pretrain_source synthetic_10class \
		--transfer_source synthetic_transfer \
		--seeds 6

cifar10-jepa:
	python scripts/run_jepa.py --source cifar10 \
		--image_size 32 --embed_dim 128 --encoder_layers 4 \
		--pretrain_epochs 100 --pretrain_batch_size 128 \
		--n_pretrain_max 50000 \
		--use_augmentation --use_block_masks \
		--n_per_class_values 10 50 200 1000 \
		--readouts linear attn_pool knn \
		--seeds 4

stl10-jepa:
	python scripts/run_jepa.py --source stl10 \
		--image_size 64 --patch_size 8 --embed_dim 192 --encoder_layers 6 \
		--n_scales 3 \
		--pretrain_epochs 200 --pretrain_batch_size 256 --pretrain_lr 1.5e-3 \
		--n_pretrain_max 100000 \
		--use_augmentation --use_block_masks \
		--n_per_class_values 10 50 200 500 \
		--readouts linear attn_pool knn \
		--fine_tune_at 500 \
		--seeds 4

clean:
	rm -rf results/* data/
