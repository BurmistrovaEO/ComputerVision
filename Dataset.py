import torchvision
from torchvision import transforms
import splitfolders

#data_folder = 'C:/Users/Kate/Documents/datasets/archive/img_align_celeba/img_align_celeba/'

#splitfolders.ratio(data_folder, output='C:/Users/Kate/Documents/datasets/archive/img_align_celeba/output', seed=1337, ratio=(.7,.0,.3))

transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
