# encoding: utf-8
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

from data.imagemod import GetTransforms
#from easydict import EasyDict as edict

class ChexpertDataset(Dataset):
    def __init__(self, conf, csvFilePath, phase = 'train', isGrayScale = False, u_approach = 0):
        """
        Args:
            conf (easyDict): Config file as easydict object.
            gray_scale (Boolean): Whether image should be treated as grayscale.

            u_approach: Approach for uncertain labels (zeroes = 0, ones = 1, ignore = -1 (, self-trained = 2)).
                                                            (are zeroes and ignores same??)
        """
        self._conf = conf
        self._csvFilePath = csvFilePath
        self._phase = phase
        self._isGrayScale = isGrayScale
        self._u_approach = u_approach
        
        self._numImages = 0
        self._classes = []      #All classes which will be used for training/testing.
        self._image_paths = []  #List of all image locations.
        self._labels = []       #List of labels for all images.

        #_approach_dict[0] - U-Zero
        # _approach_dict[1] - U-Ones        
        self._approach_dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'} ]

        image_names = []
        #TO-DO: get file names in image_name list

        with open(self._csvFilePath) as f:
            
            header = f.readline().strip('\n').split(',')
            #self._classes = header[5:]  # Train/test for all classes,
                                    # or for the 5 classes # = [header[7], header[10], header[11], header[13], header[15]]
                                    # This must match with the number of 1s in conf.num_classes
            self._classes = [header[7], header[10], header[11], header[13], header[15]]

            # Read each datapoint one line at a time
            for line in f:
                fields = line.strip('\n').split(',')    #fields: all attributes' values.
                labels = []                             #labels: only manifestation attribute values.
                flg_enhance = False
                #Ignore Lateral X-rays
                if fields[3] != "Frontal":
                    continue

                # If u-ignore (u_approach = -1), continue loop. Do not use this
                # instance to train the model.
                # TO DO: Check labels where uncertainty value is, and only
                # then ignore the row.
                if u_approach == -1 and "-1.0" in fields[5:]:
                    continue
                
                image_path = fields[0]
                
                '''
                # Read each label value and replace with corresponding value from _approach_dict
                # (For u_zeroes and u_ones. Self-train NOT IMPLEMENTED.)
                for index, i in enumerate(fields[5:]):
                    labels.append(self._approach_dict[self._u_approach].get(i))

                    #_conf.enhance_index: index of manifestations with less instances. Increase instance by _conf.enhance_times times.
                    if self._approach_dict[self._u_approach].get(i) == '1' and self._conf.enhance_index.count(index) > 0:
                        flg_enhance = True
                '''

                # Instead of using same approach for all labeles,
                # we shall be using U-Ones for Edema (5) and Atelectasis (8), and 
                # U-Zeroes for Cardiomegaly (2), Consolidation (6) and Pleural Effusion (10).
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self._approach_dict[1].get(value))
                        if self._approach_dict[1].get(value) == '1' and self._conf.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self._approach_dict[0].get(value))
                        if self._approach_dict[0].get(value) == '1' and self._conf.enhance_index.count(index) > 0:
                            flg_enhance = True
                
                # labels = ([self.dict.get(n, n) for n in fields[5:]])  #
                labels = list(map(int, labels))     # Convert strings to int.
                image_path = os.path.join(self._conf.dataset_location, "..", image_path)    # full path to image
                self._image_paths.append(image_path)
                self._labels.append(labels)
                if flg_enhance == True and self._phase == 'train':
                    for i in range(self._conf.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)

            self._numImages = len(self._image_paths)

    def __len__(self):
        return self._numImages

    def _fix_ratio(self, image):
        h, w, c = image.shape

        if h >= w:
            ratio = h * 1.0 / w
            h_ = self._conf.long_side
            w_ = round(h_ / ratio)
        else:
            ratio = w * 1.0 / h
            w_ = self._conf.long_side
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_),
                           interpolation=cv2.INTER_LINEAR)

        image = self._border_pad(image)

        return image

    def _border_pad(self, image):
        h, w, c = image.shape

        if self._conf.border_pad == 'zero':
            image = np.pad(
                image,
                ((0, self._conf.long_side - h),
                 (0, self._conf.long_side - w), (0, 0)),
                mode='constant', constant_values=0.0
            )
        elif self._conf.border_pad == 'pixel_mean':
            image = np.pad(
                image,
                ((0, self._conf.long_side - h),
                 (0, self._conf.long_side - w), (0, 0)),
                mode='constant', constant_values=self._conf.pixel_mean
            )
        else:
            image = np.pad(
                image,
                ((0, self._conf.long_side - h),
                 (0, self._conf.long_side - w), (0, 0)),
                mode=self._conf.border_pad
            )

        return image

    def __getitem__(self, index):
        '''
        image_name = self.image_names[index]
        if self.gray_scale:
            image = Image.open(image_name).convert('L')
        else:
            image = Image.open(image_name).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)
        return image
        '''
        image = cv2.imread(self._image_paths[index], 0)
        image = Image.fromarray(image)
        if self._phase == 'train':
            image = GetTransforms(image, type=self._conf.use_transforms_type)

        image = np.array(image)
        
        if self._conf.use_equalizeHist:
            image = cv2.equalizeHist(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
        
        if self._conf.fix_ratio:
            image = self._fix_ratio(image)
        else:
            image = cv2.resize(image, dsize=(self._conf.width, self._conf.height),
                               interpolation=cv2.INTER_LINEAR)
        if self._conf.gaussian_blur > 0:
            image = cv2.GaussianBlur(image, (self._conf.gaussian_blur,
                                             self._conf.gaussian_blur), 0)
        
        # Dataset mean: 0.533048452958796; dataset std 0.03490651403764978 - not using this here.
        # Normalization
        image -= self._conf.pixel_mean
        # Do not use pixel_std for VGG and ResNet; use for DenseNet and InceptionNet.
        if self._conf.use_pixel_std:
            image /= self._conf.pixel_std
        
        # normal image tensor :  H x W x C
        # torch image tensor :   C X H X W
        image = image.transpose((2, 0, 1))
        labels = np.array(self._labels[index]).astype(np.float32)

        path = self._image_paths[index]

        if self._phase == 'train' or self._phase == 'dev':
            return (image, labels)
        elif self._phase == 'test':
            return (image, path)
        elif self._phase == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown \'Phase\' value : {}'.format(self._phase))
