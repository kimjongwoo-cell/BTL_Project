1. Extract Features (if you haven't already)
python extract_features.py --classes Ref Degree Distance Particle Crack --crop --encoder resnet18

2. Train Your Model

Example (for Ref vs. Degree):
python main.py --features_dir features/resnet18 --method patch --pooling transmil --classes Ref Degree

Example (for Ref vs. Particle and Crack):
python main.py --features_dir features/resnet18 --method patch --pooling transmil --classes Ref Particle+Crack


# 비율에 맞춰서 split
python main.py --features_dir features/resnet18 --classes Ref Particle+Crack --method patch --pooling transmil  --split_method ratio

# 각 클래스에서 훈련 샘플을 40개씩 지정하여 훈련
python main.py --features_dir features/resnet18 --classes Ref Particle+Crack --method patch --pooling transmil  --split_method fixed  --train_samples_per_class 40

## Polar feature extraction (optional)

To extract features after polar transforming each image and save under `features/polar/<encoder>`:

```
python extract_features.py --data_dir ./data/측정_250926 \
	--classes Ref Degree Distance Particle Crack \
	--crop --encoder resnet50 --use_polar --r_bins 512 --s_bins 1024 --signal_ratio 0.2
```

Then train from the polar features directory, e.g.:

```
python main.py --features_dir features/polar/resnet50 --method patch --pooling transmil --classes Ref Degree
```