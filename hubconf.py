from hydra.experimental import compose, initialize
import torch
from exp.gpv.models.gpv import GPV

dependencies = ["torch", "torchvision"]


def gpv1(pretrained=False, checkpoints_path=""):
    with initialize(config_path='configs', job_name='inference'):
        cfg = compose(config_name='exp/gpv_inference')
    model = GPV(cfg.model)
    if pretrained:
        loaded_dict = torch.hub.load_state_dict_from_url(
            url="https://ai2-prior-gpv.s3-us-west-2.amazonaws.com/public/trained_models/gpv_all_original_split/ckpts/"
                "model.pth", map_location="cpu", )['model']
    elif checkpoints_path:
        loaded_dict = torch.load(checkpoints_path, map_location="cpu")['model']
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = loaded_dict[f'module.{k}']
        state_dict[k].requires_grad = False
    model.load_state_dict(state_dict)
    return model
