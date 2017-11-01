from DataAugmentation import DataAugmentation

aug_params = {
	'data_root': '../../../data/images/',
    'data_list': '../../../data/train.txt',
    'data_aug_list': '../../../data/aug_train2.txt',
    'img_dimension': 128,
    'max_rotate_angle': 10,
    'rotate_zero_padding': False,
    'random_flip': True,
    'saturation_range': (0.5, 1.5),
    'contrast_range': (0.5, 1.5),
    'brightness_range': (0.5, 1.5),
    'iteration': 11
}

data = DataAugmentation(**aug_params)
data.augmente()