from tqdm import tqdm
import os
import shutil

from load_data import load_datasets


def get_norm():
    """celeba='/data_nas/users/shx/datasets_n/celeba'
    lfw='/data_nas/users/shx/datasets_n/LFW'
    selfie='/data_nas/users/shx/datasets/DIRE/Selfie_dire'
    sfhq='/data_nas/users/shx/datasets/DIRE/sfhq_dire'
    sdface='/data_nas/users/shx/datasets/DIRE/sdfake_dire'
    save_path='/data_nas/users/shx/datasets_n'
    #paths=[celeba,lfw,selfie,sfhq,sdface]
    #types=['celeba', 'lfw', 'selfie', 'sfhq', 'sdface' ]

    df_imagenet='/data_nas/users/shx/datasets/df_imagenet'
    artifact='/data_nas/users/shx/datasets/Artifact'
    artifact_dire='/data_nas/users/shx/datasets/Artifact_dire'
    genimage='/data_nas/users/yhz/datasets/GenImage'
    genimage_dire='/data_nas/users/yhz/datasets/GenImage_dire'
    imagenet='/data/imagenet'
    df='/data_nas/users/shx/datasets/DiffusionForensics/images'
    df_dire='/data_nas/users/shx/datasets/DiffusionForensics/dire'

    paths=[df_imagenet,artifact,artifact_dire,genimage,genimage_dire,imagenet,df,df_dire]
    types=['n', 'artifact', 'artifact', 'genimage', 'genimage','n' ,'df','df']
    names=['df_imagenet','artifact','artifact_dire','genimage','genimage_dire','imagenet','df','df_dire']
    """

    save_path = "/data_nas/users/shx/datasets_n"
    paths = ["/data_nas/users/shx/datasets/Artifact"]
    types = ["artifact"]
    names = ["artifact2"]

    train_datasets, val_datasets = load_datasets(paths, types, concat=False)

    for i in range(len(train_datasets)):
        this_save_path = save_path + "/" + names[i] + "/train"
        train_datasets[i].copy_img(this_save_path)

    for i in range(len(val_datasets)):
        this_save_path = save_path + "/" + names[i] + "/test"
        val_datasets[i].copy_img(this_save_path)


def check_dataset():
    celeba = "/data_nas/users/shx/datasets_n/celeba"
    lfw = "/data_nas/users/shx/datasets_n/lfw"
    selfie = "/data_nas/users/shx/datasets_n/selfie"
    sfhq = "/data_nas/users/shx/datasets_n/sfhq"
    sdface = "/data_nas/users/shx/datasets_n/sdface"
    tail_list = ["_adv", "_adv_dire", "_dire"]
    paths1 = [celeba, lfw, selfie, sfhq, sdface]
    paths = []
    for p in paths1:
        paths.append(p)
        for t in tail_list:
            paths.append(p + t)
    types = ["n"] * len(paths)
    train_datasets, val_datasets = load_datasets(paths, types, concat=False)


from PIL import Image


def check_dataset2():
    path = "/data_nas/users/shx/datasets_n/df_imagenet"

    def load_img(dataset):
        for i in tqdm(range(len(dataset))):
            try:
                image = Image.open(dataset.image_paths[i]).convert("RGB")
            except OSError as e:
                print(f"Error loading image: {e} {dataset.image_paths[i]}")

    train_datasets, val_datasets = load_datasets([path], ["n"], concat=True)
    load_img(train_datasets)
    load_img(val_datasets)


if __name__ == "__main__":
    # get_norm()
    check_dataset2()
