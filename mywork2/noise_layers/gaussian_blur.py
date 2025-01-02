import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def gaussian_blur(image, radius, to_pil=False):
    """
    Applies a differentiable Gaussian blur to an image.

    Parameters:
    - image: PIL Image or torch.Tensor. The input image to be blurred.
    - radius: float. The radius of the blur.
    - to_pil: bool. Whether to convert the output back to a PIL Image.

    Returns:
    - Blurred image as PIL Image if to_pil is True, else as torch.Tensor.
    """
    sigma = radius / 3
    kernel_size = 2 * int(2 * sigma + 0.5) + 1

    if isinstance(image, Image.Image):
        img_tensor = transforms.ToTensor()(image)
        img_tensor = img_tensor.unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3:
            img_tensor = image.unsqueeze(0)
        elif image.ndim == 4:
            img_tensor = image
        else:
            raise ValueError("Tensor must have 3 or 4 dimensions.")
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or torch.Tensor.")

    blurred_tensor = F.gaussian_blur(img_tensor, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))

    if isinstance(image, Image.Image) or to_pil:
        blurred_tensor = blurred_tensor.squeeze(0)
        blurred_image = transforms.ToPILImage()(blurred_tensor)
        return blurred_image
    else:
        return blurred_tensor