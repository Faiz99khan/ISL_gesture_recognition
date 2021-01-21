import random
import math
import numbers
import collections
import numpy as np
import torch
import cv2
import scipy.ndimage
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


#import dlib
from facenet_pytorch import MTCNN
import cv2 

import numpy as np
from PIL  import Image
import os
from time import time
#from google.colab.patches import cv2_imshow

from torchvision.transforms.functional import adjust_brightness,adjust_contrast,perspective
from torchvision.transforms import RandomPerspective
import random

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,index=None):
        for t in self.transforms:
            if isinstance(t,(MultiScaleCornerCrop,ExtractRoi)):
                img= t(img,index)
            else:
                img = t(img)
        return img

    def randomize_parameters(self,data_size=None):
        for t in self.transforms:
          if isinstance(t,(MultiScaleCornerCrop,ExtractRoi)):
              t.randomize_parameters(data_size)
          else: 
              t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation   

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(img,np.ndarray):
          img=Image.fromarray(img)

        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class ExtractRoi(object):

    def __init__(self,no_cuda,size=(112,112)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        if no_cuda:
          device='cpu'
        else:
          device='cuda'
        self.fast_mtcnn=MTCNN(factor=0.5,device=device)
        self.roi_dict={}                    ####
        self.is_first_epoch=True
        self.is_first_time=True
        self.i=0

    def __call__(self, img,index):

        if Image.isImageType(img):
          img=np.array(img)

        if self.is_first_epoch:
            if self.is_first_sample:
                self.is_first_sample=False
                x1,y1,x2,y2=extract_roi(img,self.fast_mtcnn)              
                self.roi_dict[index]=(x1,y1,x2,y2)
            else:
                x1,y1,x2,y2=self.roi_dict[index]
        else:
             x1,y1,x2,y2=self.roi_dict[index]
        


        img=img[y1:y2,x1:x2]
        return img

    def randomize_parameters(self,data_size=None):
        self.i+=1
        if self.is_first_time:
          self.data_size=data_size
          self.is_first_time=False
        if self.i>self.data_size:
           self.is_first_epoch=False
        
        self.is_first_sample=True



class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.)) - 16  ####
            y1 = int(round((image_height - th) / 2.)) + 16   ####
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 scales,
                 size,
                 interpolation=Image.BILINEAR,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation       ####
        self.crop_positions = crop_positions
        
        self.roi_dict={}
        self.is_first_epoch=True
        self.is_first_time=True
        self.i=0

    def __call__(self, img,index):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
           
            if self.is_first_epoch:
                if self.is_first_sample:
                    self.is_first_sample=False
                    x1,y1,x2,y2=extract_roi(np.array(img))              
                    self.roi_dict[index]=(x1,y1,x2,y2)
                else:
                     x1,y1,x2,y2=self.roi_dict[index]
            else:
                x1,y1,x2,y2=self.roi_dict[index]
          

            '''center_x = image_width // 2 - 24  ####
            center_y = image_height // 2  ####
            box_half = crop_size // 2
            x1 = center_x - box_half  - 90 ####
            y1 = center_y - box_half
            x2 = center_x + box_half  
            y2 = center_y + box_half'''  
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1,y1,x2,y2))
        return img                      ####
        #return img.resize((self.size, self.size), self.interpolation)   ####

    def randomize_parameters(self,data_size):
        self.i+=1
        if self.is_first_time:
          self.data_size=data_size
          self.is_first_time=False
        if self.i>self.data_size:
           self.is_first_epoch=False
        
        self.is_first_sample=True

        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(
            0,
            len(self.scales) - 1)]
        


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        #self.interpolation = interpolation     #####
        self.interpolation = Image.BICUBIC

    def __call__(self, img):
        '''min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        img = img.crop((x1, y1, x2, y2))
        return img.resize((self.size, self.size), self.interpolation)'''

        if self.is_first_time:                                  #### my code
            self.is_first_time=False
            w,h=img.size
            yd=h-h*self.scale
            xd=w-w*self.scale
            self.x1=int(0.5*xd+random.randint(-xd//3,xd//3))
            self.y1=int(0.5*yd+random.randint(-yd//5,yd//5))
            self.x2=int(self.x1+w*self.scale)
            self.y2=int(self.y1+h*self.scale)


        img = img.crop((self.x1, self.y1, self.x2, self.y2))

        return img.resize((self.size, self.size), self.interpolation)


    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        #self.scale = 1
#        self.tl_x = random.random()
#        self.tl_y = random.random()
        self.is_first_time=True                 #####



class SpatialElasticDisplacement(object):

    def __init__(self, sigma=2.0, alpha=1.0, order=3, cval=0, mode="constant"):         #### sigma=3
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, img):
        if self.p < 0.65:
            is_PIL = isinstance(img, Image.Image)
            if is_PIL:
                img = np.asarray(img)

            image = img
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            ret_image = (self._map_coordinates(
                image,
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))
            import matplotlib.pyplot as plt
#            plt.imshow(ret_image)
            if is_PIL:
                return Image.fromarray(ret_image)
            else:
                return ret_image
        else:
            return img

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2),"shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3),"image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result

    def randomize_parameters(self):
       self.p = random.random()


class RandomRotate(object):

    def __init__(self,interpolation=Image.BICUBIC):
        self.interpolation = interpolation
    def __call__(self, img):
        im_size = img.size
        ret_img = img.rotate(self.rotate_angle, resample=self.interpolation)
#        plt.imshow(ret_img)
        return ret_img

    def randomize_parameters(self):
        rint=random.randint(0,1)
        if rint:
           self.rotate_angle = random.randint(-7, 7)
        else:
            self.rotate_angle=0


class RandomResize(object):

    def __init__(self,interpolation=Image.BICUBIC):
        self.interpolation = interpolation
    def __call__(self, img):
        if isinstance(img,np.ndarray):
          img=Image.fromarray(img)
        
        im_size = img.size
        ret_img = img.resize((int(im_size[0]*self.resize_const),
                              int(im_size[1]*self.resize_const)),resample=self.interpolation)

        return ret_img

    def randomize_parameters(self):
        self.resize_const = random.uniform(0.9, 1.1)



class Gaussian_blur(object):

    def __init__(self, radius=0.0):
        self.radius = radius

    def __call__(self, img):
        if self.p < 0.2:
            blurred = ndimage.gaussian_filter(img, sigma=(5, 5, 0), order=0)
            return blurred
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        self.radius = random.uniform(0.0, 0.1)


class SaltImage(object):
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, img):
        is_PIL = isinstance(img, Image.Image)
        if is_PIL:
            img = np.asarray(img)

        if self.p < 0.10:
            data_final = []
            img = img.astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)

            if is_PIL:
                return Image.fromarray(img.astype(np.uint8))
            else:
                return img
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        self.ratio = random.randint(80, 120)


class Dropout(object):

    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, img):
        is_PIL = isinstance(img, Image.Image)
        if is_PIL:
            img = np.asarray(img)

        if self.p < 0.10:
            data_final = []
            img = img.astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            if is_PIL:
                return Image.fromarray(img.astype(np.uint8))
            else:
                return img
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        self.ratio = random.randint(30, 50)


class MultiplyValues():

    def __init__(self, value=0.2, per_channel=False):
        self.value = value
        self.per_channel = per_channel

    def __call__(self, img):
        is_PIL = isinstance(img, Image.Image)
        if is_PIL:
            img = np.asarray(img)

        image = img.astype(np.float64)
        image *= self.sample
        image = np.where(image > 255, 255, image)
        image = np.where(image < 0, 0, image)
        image = image.astype(np.uint8)

        if is_PIL:
            return Image.fromarray(image)
        else:
            return image

    def randomize_parameters(self):
        self.sample = random.uniform(1.0 - self.value, 1.0 + self.value)

class FinalSize():
  def __init__(self,size=(112,112),interpolation=Image.BICUBIC):
    if isinstance(size,tuple):
      self.size=size
    else:
      self.size=(size,size)
    self.interpolation=interpolation          
  def __call__(self,img):
      if isinstance(img,np.ndarray):
        img=Image.fromarray(img)
      return img.resize(self.size,self.interpolation)
  
  def randomize_parameters(self):
      pass

class RandomBrightnessContrast():

  def __call__(self,img):
      if not (isinstance(img,torch.Tensor) or Image.isImageType(img)):
        img=Image.fromarray(img)
      img = adjust_brightness(img,self.bf)
      return adjust_contrast(img,self.cf)
  
  def randomize_parameters(self):
      self.bf=random.uniform(.7,1.3)
      self.cf=random.uniform(.7,1.3)

class Random_Perspective():
    def __init__(self,size=(112,112)):
        self.size=size
        self.distortion=.3
        self.rpersp=RandomPerspective(self.distortion)   

    def __call__(self,img):
        if not (isinstance(img,(torch.Tensor,Image.Image)) or Image.isImageType(img)):
          img=Image.fromarray(img)
        if self.skip:
          return img
        else:
          return perspective(img,self.sp,self.ep)

    def randomize_parameters(self):
        rint=np.random.choice([0,1],p=[.65,.35])
        if rint:
          self.sp,self.ep=self.rpersp.get_params(self.size[0],self.size[1],self.distortion)        
          self.skip=False
        else:
          self.skip=True
          
class InferenceExtractRoi(object):

    def __init__(self,no_cuda,only_cordinates=True,interpolation=Image.BICUBIC):

        self.interpolation=interpolation
        if no_cuda:
          device='cpu'
        else:
          device='cuda'
        self.fast_mtcnn=MTCNN(factor=.5,device=device)
        self.only_cordinates=only_cordinates

    def __call__(self,img):
        if Image.isImageType(img):
          img=np.array(img)

        self.x1,self.y1,self.x2,self.y2=extract_roi(img,self.fast_mtcnn)              

        if self.only_cordinates:
          return self.x1,self.y1,self.x2,self.y2
        else:
          return img[ self.y1: self.y2,self.x1:self.x2]

    def randomize_parameters(self):
        pass
            
def extract_roi(img,fast_mtcnn=None):
    div=min(img.shape[0],img.shape[1])//108
    while True:
        rsz_img=cv2.resize(img,(img.shape[1]//div,img.shape[0]//div))
        #s=time()
        rects = fast_mtcnn.detect(rsz_img)
        #print(time()-s)        
        if rects[0] is not None and len(rects[0])==1:
          boxes=[]
          for (i, rect) in enumerate(rects[0]):
            #print('confidence',rects[1][0])
            x1,y1,x2,y2=rect
            box=(x1,y1,x2,y2)
            boxes.append(box)
            face=boxes
          break
        
        if div<2:
            #print('unable to find face')
            break
        div-=1

    if rects[0] is not None and len(rects[0])==1:
        x1,y1,x2,y2=face[0]
        w,h=x2-x1,y2-y1
        #print('\t\t\t',w,h)
        xc,yc=int((x1+w/2)*div),int((y1+h/2)*div)
        w2,h2=int(w*div),int(h*div)
        x1=int(xc-w2*3.2)      # max 3.2  for fast mtcnn
        y1=int(yc-h2*.7)      # max 0.7 for fast mtcnn
        x2=int(xc+w2*1.6)      # max 1.6 for fast mtcnn
        y2=int(yc+h2*2.6)      # max 2.6  for fast mtcnn
        if x1<0: x1=0
        if y1<0: y1=0
        if x2>img.shape[1]: x2=img.shape[1]
        if y2>img.shape[0]:  y2=img.shape[0]
        

    else:
       # print('..................................here no face is found')
        h,w,_=img.shape
        crp_size=min(h,w)
        half_box=crp_size//2
        yc,xc=h//2,w//2-24
        x1=xc-half_box -110
        y1=yc-half_box
        x2=xc+half_box + 30
        y2=yc+half_box

        if x1<0: x1=0
        if y1<0: y1=0
        if x2>img.shape[1]: x2=img.shape[1]
        if y2>img.shape[0]:  y2=img.shape[0]
        
   # print(itr,div,w,h,w2,h2)#,w2,h2)
    return (x1,y1,x2,y2)
