from PIL import Image


def to_tif(path_or_image, save_path="/home/maxiaohui/box_muller_wm/MaskedDenoising/testset/McM/truth", name="output_image"):
    file_name = None
    if isinstance(path_or_image, str):
        img = Image.open(path_or_image)
        file_name = path_or_image.split("/")[-1].split(".")[0]
    else:
        img = path_or_image
        file_name = name
    save_path = save_path + "/" + file_name
    img.save(save_path + ".tif", format="TIFF")
    

if __name__=="__main__":
    path = "/home/maxiaohui/box_muller_wm/MaskedDenoising/testset/McM/test/fish.JPEG"
    to_tif(path)