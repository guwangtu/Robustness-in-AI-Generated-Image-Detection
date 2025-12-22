# Robustness-in-AI-Generated-Image-Detection

conda env create -f environment.yml -n ragid

conda activate ragid

# Dataset format

dataset/
└── celeba/
    ├── train/
    │   ├── fake/
    │   │   ├── image_001.jpg
    │   │   ├── image_002.jpg
    │   │   └── ...
    │   └── real/
    │       ├── image_001.jpg
    │       ├── image_002.jpg
    │       └── ...
    └── test/
        ├── fake/
        │   ├── image_001.jpg
        │   └── ...
        └── real/
            ├── image_001.jpg
            └── ...

# test use DRR

python main.py --todo test  --simple_test --load_path checkpoints/dire_face_celeba_sfhq_adv.pt --device 0 --data_paths datasets/sfhq_pgd_multieps_dire/2pgd_16/test

python main.py --todo test  --simple_test --load_path checkpoints/dire_face_celeba_sfhq_adv.pt --device 0 --data_paths datasets/celeba_pgd_multieps_dire/2pgd_16/test