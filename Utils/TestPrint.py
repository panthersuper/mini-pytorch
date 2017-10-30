import scipy.misc
import numpy as np
import math
import torch
from torch.autograd import Variable

class TestPrint(object):
    def __init__(self, **kwargs):
        self.data_root = kwargs['data_root']
        self.data_result_list = kwargs['data_result_list']
        self.test_num = kwargs['test_num']
        self.load_size = kwargs['load_size']
        self.fine_size = kwargs['fine_size']
        self.data_mean = kwargs['data_mean']
        self.model = kwargs['model']

    def PrintToFile(self):
        with open(self.data_result_list, 'w') as text_file:
            text_file.write('')

        for i in range(self.test_num):         
            with open(self.data_result_list, 'a') as text_file:
                newpath = 'test/'+'{:08.0f}'.format(i+1)+'.jpg'
                text_file.write(newpath)

                image = scipy.misc.imread(self.data_root+newpath)
                image = scipy.misc.imresize(image, (self.load_size, self.load_size))
                image = image.astype(np.float32)/255.
                image = image - self.data_mean

                offset_h = math.floor((self.load_size-self.fine_size)/2)
                offset_w = math.floor((self.load_size-self.fine_size)/2)

                images_batch = np.zeros((1, self.fine_size, self.fine_size, 3)) 
                images_batch[0, ...]  =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
                images_batch = np.swapaxes(images_batch,1,3)
                images_batch = np.swapaxes(images_batch,2,3)
                images = torch.from_numpy(images_batch).float().cuda()

                outputs = self.model(Variable(images))
                _, predicted = torch.topk(outputs.data, 5)

                for i in range(5):
                    text_file.write(' '+str(predicted[0][i]))

                text_file.write('\n')
                