from load_data import load_image_fold

path = "/data_nas/users/shx/datasets/DiffusionForensics/image_test/test/image"
image = load_image_fold(path)
print(len(image))
