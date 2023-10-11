import numpy as np 
from PIL import Image
from pathlib import Path
import torchvision


# First download dataset from https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip
# Extract into ../data so that path is ../data/kvasir-dataset-v2
# Should contain eight folders

def get_ccrop(width, height):
    # width always geq height for Kvasir
    # crop to 5/4 width to height ratio
    if width / height > 1.25:
        # need to crop width dimension
        crop_size = 1.25 * height
        # torchvision crop expects (H, W) order
        ccrop = torchvision.transforms.CenterCrop((height, crop_size))
    elif width / height < 1.25:
        # need to crop height dimension
        crop_size = width / 1.25
        # torchvision crop expects (H, W) order
        ccrop = torchvision.transforms.CenterCrop((crop_size, width))
    else:
        ccrop = torchvision.transforms.CenterCrop((height, width)) # nop
    
    return ccrop


def get_image(filename):
    im = Image.open(filename)
    width, height = im.size
    ccrop = get_ccrop(width, height)
    cropped_im = ccrop(im)
    # rescale image to 100 width by 80 height
    final_im = torchvision.transforms.Resize([80, 100])(cropped_im)
    im_data = np.array(final_im.getdata()).reshape(80, 100, 3)
    im.close()
    return im_data


# Run data preprocessing of Kvasir to create kvasir.npz archive
if __name__ == "__main__":
    folder_names = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
    data_root = Path('../data/kvasir-dataset-v2')
    
    images_list = []
    labels_list = []
    for i, folder in enumerate(folder_names):
        print(f"Parsing {folder}, {i+1} of 8")
        folder_path = data_root / folder

        filenames = list(folder_path.glob(r'**/*.jpg'))
        for filename in filenames[:]:
            im = get_image(filename)
            images_list.append(im)
            labels_list.append(i)
        
    ims = np.stack(images_list, axis=0)
    labels = np.stack(labels_list, axis=0)

    np.savez("../data/kvasir.npz", images=ims, labels=labels)