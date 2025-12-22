import argparse


def str_to_float(value_str):
    """
    将分数形式的字符串转换为浮点数。
    例如: "8/255" -> 0.03137254901960784
    """
    if "/" in value_str:
        numerator, denominator = map(float, value_str.split("/"))
        return numerator / denominator
    else:
        return float(value_str)


def parser():
    parser = argparse.ArgumentParser(description="attack")

    parser.add_argument(
        "--todo",
        choices=["train", "test", "degrade", "get_imgs"],
        default="train",
        help="train|test|degrade|get_imgs",
    )
    parser.add_argument("--device", default="0", type=str, help="0123")

    parser.add_argument("--model", default="resnet", type=str, help="resnet,vit")

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train_dataset2", default=None)
    parser.add_argument("--val_dataset2", default=None)
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--adv_test", default=False, action="store_true")
    parser.add_argument("--not_shuffle", default=False, action="store_true")
    parser.add_argument("--save_each_epoch", type=int, default=1)
    parser.add_argument("--save_path", default="test")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--load_path2", default=None)
    parser.add_argument("--load_epoch", type=int, default=0)
    parser.add_argument("--sgd", default=False, action="store_true")
    parser.add_argument("--test_each_batch", type=int, default=0)
    parser.add_argument("--save_loss", default=False, action="store_true")

    parser.add_argument(
        "--diffusion_path",
        type=str,
        default="/data/user/shx/Generate_image_detection/compute_dire/models/256x256_diffusion_uncond.pt",
    )
    parser.add_argument("--real_path", default=None)
    parser.add_argument("--fake_path", default=None)
    parser.add_argument("--each_class_num", type=int, default=0)

    parser.add_argument("--adv", default=False, action="store_true")
    parser.add_argument(
        "--adv_mode", type=int, default=0
    )  # 0:normal 1:Manually implemented 2:autoattack 3:DDN 4:CW 5:quareAttack l2 6:quareAttack linf
    parser.add_argument("--CW_c", type=float, default=0.0001)  # CW攻击的强度系数


    parser.add_argument("--diff_denoise", default=False, action="store_true")
    parser.add_argument("--diff_denoise_test", default=False, action="store_true")
    # parser.add_argument("--denoise_train_beta", type = float, default = 1.0)
    parser.add_argument("--diff_denoise_t", type=int, default=2)
    parser.add_argument("--atk_eps", type=str_to_float, default=8 / 255)  # bound
    parser.add_argument("--atk_alpha", type=str_to_float, default=2 / 255)
    parser.add_argument("--atk_steps", type=int, default=10)
    parser.add_argument("--update_adv_each_epoch", type=int, default=100)

    parser.add_argument("--TRADES", default=False, action="store_true")
    parser.add_argument("--TRADES_beta", type=float, default=1.0)
    parser.add_argument("--MART", default=False, action="store_true")
    parser.add_argument("--MART_alpha", type=float, default=0.2)
    parser.add_argument("--MART_beta", type=float, default=6.0)
    parser.add_argument("--normal_adv", default=False, action="store_true")

    parser.add_argument(
        "--save_denoise_pic_training", default=False, action="store_true"
    )
    parser.add_argument("--save_denoise_path", default=None)

    parser.add_argument("--test_first", default=False, action="store_true")
    parser.add_argument("--artifact", default=False, action="store_true")
    parser.add_argument("--df", default=False, action="store_true")
    parser.add_argument("--genimage", default=False, action="store_true")
    parser.add_argument("--imagenet", default=None)

    parser.add_argument(
        "--data_paths", type=str, nargs="+", default=None, help="Paths to the datasets"
    )
    parser.add_argument(
        "--data_types",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "laion",
            "celeba",
            "lfw",
            "selfie",
            "sfhq",
            "sdface",
            "artifact",
            "df",
            "genimage",
            "imagenet",
            "fold",
            "n",
        ],
        help="Types of the datasets",
    )
    parser.add_argument("--n", default=False, action="store_true")

    parser.add_argument("--ratio_list", type=float, nargs="+", default=None)
    parser.add_argument("--validation_split", type=float, default=0.2)

    parser.add_argument("--val_rate", type=float, default=1.0)
    parser.add_argument("--simple_test", default=False, action="store_true")
    # parser.add_argument("--log_path", default="log/training.log") #改成自动的了

    return parser.parse_args()
