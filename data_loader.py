
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split

from helper_code import *
#===================================
# Parameters (configure stuff here)
#===================================

data_transform = {
    "train": transforms.Compose([transforms.Resize((425,550)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize((425,550)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "test": transforms.Compose([transforms.Resize((425,550)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

#=========
# Classes
#=========

class MyDataSet(Dataset):
    def __init__(self, image_paths: list, image_classes: list, transform=None):
        self.image_paths = image_paths
        self.image_classes = image_classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item])

        if img.mode != 'RGB':
            img = img.convert("RGB")

        label = self.image_classes[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



def get_training_and_validation_loaders(list_of_all_labels, image_path_list, label_names_list):
    """
    Given  a list `list_of_all_labels` of all class labels in the dataset,
    `image_path_list` (a list of image paths), and `label_names_list` (a list
    of _lists_ of label names corresponding to the images), return the pair
    `(training_loader, validation_loader)` which can be used to train/validate
    a model.

    """

    #FIXME: standardize "classes" / "labels"
    num_classes = len(list_of_all_labels)

    # index --> string
    idx_to_label=list(list_of_all_labels)

    # string --> index
    label_to_idx=dict()
    for i in range(num_classes):
        label_to_idx[idx_to_label[i]]=i

    # Our dataset will encode each label not as a list of strings but as a vector,
    # one index for each class. Thus we have to convert a list like
    #   ["THING1", "THING2", "THING3"]
    # into a vector like 
    #   [0,1,0,1,1,0].

    label_list=list() # list of vectors of length len(list_of_all_labels)

    for labels in label_names_list:
        # Our vector for this index
        label_vector=[0] * num_classes
        for l in labels:
            label_vector[label_to_idx[l]] = 1
        label_list.append(label_vector)

    # Divide the dataset into training and validation sets
    training_images, validation_images, \
    training_classes, validation_classes \
        = train_test_split(image_path_list,
                           label_list, 
                           test_size=0.2,
                           random_state=42,
                           shuffle=True)  

    # Dataset for training
    train_dataset=MyDataSet(image_paths=training_images,
                            image_classes=training_classes,
                            transform=data_transform['train'])

    # Dataset for validation
    validation_dataset=MyDataSet(image_paths=validation_images,
                                 image_classes=validation_classes,
                                 transform=data_transform['val'])

    # DataLoader for training
    training_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=4,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=8,
                                                  drop_last=True,
                                                  collate_fn=train_dataset.collate_fn)

    # DataLoader for validation
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=4,
                                                    shuffle=True, # note this differs
                                                    pin_memory=True,
                                                    num_workers=8,
                                                    drop_last=True,
                                                    collate_fn=validation_dataset.collate_fn)
    
    return training_loader, validation_loader

