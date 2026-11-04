import numpy as np
from PIL import Image
import torch


def load_pair(row, width, height, device):
    """Apply one common fit rectangle to the actual generated canvas pair; no crop."""
    with Image.open(row["target_image"]) as target, Image.open(row["source_image"]) as source:
        scale = min(width / target.width, height / target.height)
        w, h = max(1, round(target.width * scale)), max(1, round(target.height * scale))
        left, top = (width-w)//2, (height-h)//2
        images = []
        for im in (source, target):
            canvas = Image.new("RGB", (width, height))
            canvas.paste(im.convert("RGB").resize((w,h), Image.Resampling.LANCZOS), (left,top))
            images.append(torch.from_numpy(np.array(canvas)).permute(2,0,1)[None]
                          .to(device=device, dtype=torch.float32) / 127.5 - 1)
    mask = torch.zeros(1,1,height,width, device=device)
    mask[:,:,top:top+h,left:left+w] = 1
    return (*images, mask, (left,top,left+w,top+h))


def save_image(path, tensor):
    array = ((tensor.detach().float().cpu()[0].permute(1,2,0).clamp(-1,1)+1)*127.5).round().numpy().astype(np.uint8)
    Image.fromarray(array).save(path)
