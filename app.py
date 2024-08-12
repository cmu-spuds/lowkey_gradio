import gradio as gr
import torch
from PIL import Image
import numpy as np
from util.feature_extraction_utils import normalize_transforms
from util.attack_utils import Attack
from util.prepare_utils import prepare_models, prepare_dir_vec, get_ensemble
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import torchvision.transforms as transforms
import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()

eps = 0.05
n_iters = 50
input_size = [112, 112]
attack_type = "lpips"
c_tv = None
c_sim = 0.05
lr = 0.0025
net_type = "alex"
noise_size = 0.005
n_starts = 1
kernel_size_gf = 7
sigma_gf = 3
combination = True
using_subspace = False
V_reduction_root = "./"
model_backbones = ["IR_152", "IR_152", "ResNet_152", "ResNet_152"]
model_roots = [
    "https://github.com/cmu-spuds/lowkey_gradio/releases/download/weights/Backbone_IR_152_Arcface_Epoch_112.pth",
    "https://github.com/cmu-spuds/lowkey_gradio/releases/download/weights/Backbone_IR_152_Cosface_Epoch_70.pth",
    "https://github.com/cmu-spuds/lowkey_gradio/releases/download/weights/Backbone_ResNet_152_Arcface_Epoch_65.pth",
    "https://github.com/cmu-spuds/lowkey_gradio/releases/download/weights/Backbone_ResNet_152_Cosface_Epoch_68.pth",
]
direction = 1
crop_size = 112
scale = crop_size / 112.0

for root in model_roots:
    torch.hub.load_state_dict_from_url(root, map_location="cpu", progress=True)


@spaces.GPU(duration=120)
def execute(attack, tensor_img, dir_vec):
    return attack.execute(tensor_img, dir_vec, direction).detach().cpu()


def protect(img, progress=gr.Progress(track_tqdm=True)):
    models_attack, V_reduction, dim = prepare_models(
        model_backbones,
        input_size,
        model_roots,
        kernel_size_gf,
        sigma_gf,
        combination,
        using_subspace,
        V_reduction_root,
    )

    img = Image.fromarray(img)
    reference = get_reference_facial_points(default_square=True) * scale
    h, w, c = np.array(img).shape

    _, landmarks = detect_faces(img)
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]

    _, tfm = warp_and_crop_face(
        np.array(img), facial5points, reference, crop_size=(crop_size, crop_size)
    )

    # pytorch transform
    theta = normalize_transforms(tfm, w, h)
    tensor_img = to_tensor(img).unsqueeze(0).to(device)

    V_reduction = None
    dim = 512

    # Find gradient direction vector
    dir_vec_extractor = get_ensemble(
        models=models_attack,
        sigma_gf=None,
        kernel_size_gf=None,
        combination=False,
        V_reduction=V_reduction,
        warp=True,
        theta_warp=theta,
    )
    dir_vec = prepare_dir_vec(dir_vec_extractor, tensor_img, dim, combination)

    img_attacked = tensor_img.clone()
    attack = Attack(
        models_attack,
        dim,
        attack_type,
        eps,
        c_sim,
        net_type,
        lr,
        n_iters,
        noise_size,
        n_starts,
        c_tv,
        sigma_gf,
        kernel_size_gf,
        combination,
        warp=True,
        theta_warp=theta,
        V_reduction=V_reduction,
    )
    img_attacked = execute(attack, tensor_img, dir_vec)

    img_attacked_pil = transforms.ToPILImage()(img_attacked[0])
    return img_attacked_pil


gr.Interface(
    fn=protect,
    inputs=gr.components.Image(height=512, width=512),
    outputs=gr.components.Image(type="pil"),
    allow_flagging="never",
).launch(show_error=True, quiet=False, share=False)
