import folder_paths
import os
import torch
import torch.nn.functional as F
from .architecture.rrdb import RRDBNet
from .architecture.grl import GRL
import comfy.model_management as mm
import comfy.utils
from contextlib import nullcontext

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
def load_rrdb(generator_weight_PATH, scale, print_options=False):  
    ''' A simpler API to load RRDB model from Real-ESRGAN
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int): the scaling factor
        print_options (bool): whether to print options to show what kinds of setting is used
    Returns:
        generator (torch): the generator instance of the model
    '''  

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

    # Find the generator weight
    if 'params_ema' in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g['params_ema']
        generator = RRDBNet(3, 3, scale=scale)    # Default blocks num is 6     

    elif 'params' in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g['params']
        generator = RRDBNet(3, 3, scale=scale)          

    elif 'model_state_dict' in checkpoint_g:
        # For my personal trained weight
        weight = checkpoint_g['model_state_dict']
        generator = RRDBNet(3, 3, scale=scale)          

    else:
        print("This weight is not supported")
        os._exit(0)


    # Handle torch.compile weight key rename
    old_keys = [key for key in weight]
    for old_key in old_keys:
        if old_key[:10] == "_orig_mod.":
            new_key = old_key[10:]
            weight[new_key] = weight[old_key]
            del weight[old_key]

    generator.load_state_dict(weight)
    generator = generator.eval()


    # Print options to show what kinds of setting is used
    if print_options:
        if 'opt' in checkpoint_g:
            for key in checkpoint_g['opt']:
                value = checkpoint_g['opt'][key]
                print(f'{key} : {value}')

    return generator

def load_grl(generator_weight_PATH, scale=4):
    ''' A simpler API to load GRL model
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int):        Scale Factor (Usually Set as 4)
    Returns:
        generator (torch): the generator instance of the model
    '''

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

     # Find the generator weight
    if 'model_state_dict' in checkpoint_g:
        weight = checkpoint_g['model_state_dict']

        # GRL tiny model (Note: tiny2 version)
        generator = GRL(
            upscale = scale,
            img_size = 64,
            window_size = 8,
            depths = [4, 4, 4, 4],
            embed_dim = 64,
            num_heads_window = [2, 2, 2, 2],
            num_heads_stripe = [2, 2, 2, 2],
            mlp_ratio = 2,
            qkv_proj_type = "linear",
            anchor_proj_type = "avgpool",
            anchor_window_down_factor = 2,
            out_proj_type = "linear",
            conv_type = "1conv",
            upsampler = "nearest+conv",     # Change
        )

    else:
        print("This weight is not supported")
        os._exit(0)


    generator.load_state_dict(weight)
    generator = generator.eval()


    num_params = 0
    for p in generator.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")


    return generator

class APISR_upscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("upscale_models"), ),
            "images": ("IMAGE",),
            "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
            "dtype": (
                    [
                        'fp32',
                        'fp16',
                    ], {
                        "default": 'fp32'
                    }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "upscale"
    CATEGORY = "ASPIR"

    def upscale(self, ckpt_name, dtype, images, per_batch):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model_path = folder_paths.get_full_path("upscale_models", ckpt_name)
        custom_config = {
            'dtype': dtype,
            'ckpt_name': ckpt_name,
        }
        
        dtype = (convert_dtype(dtype))
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.model = None
            self.current_config = custom_config
            if "RRDB" in ckpt_name:
                self.model = load_rrdb(model_path, scale=2)
            elif "GRL" in ckpt_name:
                self.model = load_grl(model_path, scale=4)
            self.model = self.model.to(dtype).to(device)
   
        images = images.permute(0, 3, 1, 2)
        B, C, H, W = images.shape
        H = (H // 8) * 8
        W = (W // 8) * 8

        if images.shape[2] != H or images.shape[3] != W:
            images = F.interpolate(images, size=(H, W), mode="bilinear")
        images = images.to(device = device, dtype = dtype)
        self.model.to(device)
        pbar = comfy.utils.ProgressBar(B)
        t = []
        autocast_condition = not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for start_idx in range(0, B, per_batch):
                sub_images = self.model(images[start_idx:start_idx+per_batch])
                t.append(sub_images.cpu())
                # Calculate the number of images processed in this batch
                batch_count = sub_images.shape[0]
                # Update the progress bar by the number of images processed in this batch
                pbar.update(batch_count)
        self.model.to(offload_device)
        
        t = torch.cat(t, dim=0).permute(0, 2, 3, 1).cpu().to(torch.float32)
        
        return (t,)
        

NODE_CLASS_MAPPINGS = {
    "APISR_upscale": APISR_upscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "APISR_upscale": "APISR Upscale",
}
