import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class ImageAugmentor(object):

    def __init__(self, param):

        self.augmentation_methods = []
        self.aug_method = param

        if self.aug_method["crop"]:
            self.augmentation_methods.append(iaa.Crop(percent=self.aug_method["crop"], sample_independently=True))
            iaa.Crop()
        if self.aug_method["rotate"]:
            self.augmentation_methods.append(iaa.Affine(rotate=self.aug_method["rotate"]))
        if self.aug_method["blur"]:
            self.augmentation_methods.append(iaa.OneOf([iaa.GaussianBlur((0, 1.3)),
                                                        iaa.AverageBlur(k=(2, 5))]))
        if self.aug_method["motion"]:
            self.augmentation_methods.append(iaa.MotionBlur(k=self.aug_method["motion"]["k"],
                                                            angle=self.aug_method["motion"]["angle"],
                                                            direction=self.aug_method["motion"]["direction"]))
        if self.aug_method["gamma"]:
            self.augmentation_methods.append(iaa.GammaContrast(gamma=self.aug_method["gamma"]))
        if self.aug_method["flip"]:
            self.augmentation_methods.append(iaa.Fliplr(p=0.5))
            self.augmentation_methods.append(iaa.Flipud(p=0.5))

        self.augmentation = iaa.Sequential(self.augmentation_methods, random_order=False)

    def transform_seg(self, img, msk):

        image = img.copy()
        mask = msk.copy()
        flag = np.sum(mask)
        # self.augmentation_methods.clear()
        segmap = SegmentationMapsOnImage(mask, shape=image.shape)
        augmentation = self.augmentation.to_deterministic()
        image, mask = augmentation(image=image, segmentation_maps=segmap)
        mask = mask.draw(size=image.shape[:2])
        mask = np.where(mask[0] > 0, 1, 0)
        mask = np.array(mask[:, :, 0], dtype=np.float64)
        auged_flag = np.sum(mask)

        if flag != 0 and auged_flag == 0:  # 确认增强后目标位置是否偏出图像范围
            msk = np.where(msk > 0, 1, 0)
            return img, msk

        return image, mask
