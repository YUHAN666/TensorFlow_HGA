from PIL import Image


def concatImage(images, mode="L"):
    if not isinstance(images, list):
        raise Exception('images must be a  list  ')
    count = len(images)
    size = Image.fromarray(images[0]).size
    target = Image.new(mode, (size[0] * count, size[1] * 1))
    for i in range(count):
        image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
        target.paste(image, (i * size[0], 0, (i + 1) * size[0], size[1]))
    return target
