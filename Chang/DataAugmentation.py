# Chang 10/28/2017

import os
import random
import math
from PIL import Image, ImageOps, ImageEnhance

class DataAugmentation(object):
    def __init__(self, **kwargs):
        self.data_root = os.path.join(kwargs['data_root'])
        self.data_list = kwargs['data_list']
        self.data_aug_list = kwargs['data_aug_list']
        self.img_dimension = kwargs['img_dimension']
        self.max_rotate_angle = kwargs['max_rotate_angle']
        self.rotate_zero_padding = kwargs['rotate_zero_padding']
        self.random_flip = kwargs['random_flip']
        self.saturation_range = kwargs['saturation_range']
        self.contrast_range = kwargs['contrast_range']
        self.brightness_range = kwargs['brightness_range']
        self.iteration = kwargs['iteration']

    def rotate(self, img, zero_padding=True, mode=Image.BICUBIC, expand=False):
        angle = random.random() * self.max_rotate_angle * random.choice([-1, 1])
        img = img.rotate(angle, mode, expand)
        if zero_padding == False:
        	offset = round(self.img_dimension * abs(math.sin(math.radians(angle))))
        	img = img.crop((offset, offset, img.size[0]-offset, img.size[1]-offset))
        return img

    def color(self, img):
        factor = random.uniform(self.saturation_range[0], self.saturation_range[1])
        img = ImageEnhance.Color(img).enhance(factor)
        factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        img = ImageEnhance.Contrast(img).enhance(factor)
        factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        img = ImageEnhance.Brightness(img).enhance(factor)
        return img

    def flip(self, img):
        factor = random.choice([0, 1])
        if factor == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def augmente(self):
        with open(self.data_aug_list, "w") as text_file:
            text_file.write('')
        num = 0

        with open(self.data_list, 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                file_path = os.path.join(self.data_root, path)
                with open(self.data_aug_list, "a") as text_file:
                    text_file.write(path + ' ' + lab + '\n')

                for i in range(self.iteration):
                	img = Image.open(file_path)
                	if self.random_flip:
                		img = self.flip(img)
                	img = self.color(img)
                	img = self.rotate(img, self.rotate_zero_padding)

                	newpath, _ = file_path.split('.jpg')
                	newpath = newpath + '_' + str(i) + '.jpg'
                	img.save(newpath)

                	with open(self.data_aug_list, "a") as text_file:
                		newpath, _ = path.split('.jpg')
                		newpath = newpath + '_' + str(i) + '.jpg'
                		text_file.write(newpath + ' ' + lab + '\n')

                num += 1
                if num % 1000 == 0:
                	print(num, "pictures done.")

        return None

