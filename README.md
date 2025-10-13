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