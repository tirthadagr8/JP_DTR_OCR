import xml.etree.ElementTree as ET
from PIL import Image
import os
from time import sleep
from tqdm import tqdm
import pandas as pd
import unicodedata
from transformers import MBart50TokenizerFast,ViTImageProcessor
from torch.utils.data import Dataset,DataLoader
from processor import *

# Paths
xml_dir = 'C:/Users/tirth/Downloads/Manga109/Manga109_released_2023_12_07/annotations/'  # Directory containing XML files
image_dir = 'C:/Users/tirth/Downloads/Manga109/Manga109_released_2023_12_07/images/'  # Directory containing page images

def convert_fullwidth_to_halfwidth(text):
    normalized_text = unicodedata.normalize('NFKC', text)
    return ''.join(c for c in unicodedata.normalize('NFD', normalized_text) if unicodedata.category(c) != 'Mn')

# Helper function to convert bounding box to x1, y1, x2, y2, x3, y3, x4, y4
def bbox_to_coordinates(xmin, ymin, xmax, ymax):
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

def load_data():
    images=[]
    annotations = []
    labels=[]

    # Iterate over all XML files
    for xml_file in tqdm(os.listdir(xml_dir)):
        if not xml_file.endswith('.xml'):
            continue
        # print(xml_file)
        # sleep(5)
        # Parse the XML file
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        
        # Process each page in the XML
        for page in root.find('pages'):
            page_index = page.get('index')
            
            # Load the corresponding image for the page
            page_image_path = os.path.join(image_dir, f'{xml_file[:-4]}/{str(page_index).zfill(3)}.jpg')
            if not os.path.exists(page_image_path):
                print(f"Warning: Image for page {page_index} not found.")
                continue
            
            # Open the page image
            # page_image = Image.open(page_image_path)
            
            # Prepare data for the first model (page-level data with annotations)
            for text in page.findall('text'):
                # Extract text annotation coordinates
                xmin, ymin = int(text.get('xmin')), int(text.get('ymin'))
                xmax, ymax = int(text.get('xmax')), int(text.get('ymax'))
                coordinates = (xmin, ymin, xmax, ymax) #bbox_to_coordinates(xmin, ymin, xmax, ymax)
                images.append(page_image_path)
                annotations.append(coordinates)
                labels.append(convert_fullwidth_to_halfwidth(text.text.strip()) if text.text else "")
        
    # pd.DataFrame({'images':images,'annotations':annotations,'labels':labels}).to_csv(os.path.join(os.getcwd(),'model_abinet_metadata.csv'))
            
    return images,labels,annotations

class custom_dataset(Dataset):
    def __init__(self,images,labels,annotations) -> None:
        super().__init__()
        self.processor=DTrOCRProcessor(DTrOCRConfig(),True,True)
        self.max_length=64
        self.images=images
        self.labels=labels
        self.annotations=annotations
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img=Image.open(self.images[index]).convert('RGB').crop(
            self.annotations[index]
        )
        inputs=self.processor(img,self.labels[index],True,padding='max_length')
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.labels[0]
        }
    
if __name__=='__main__':
    batch_size=1
    images,labels,annotations=load_data()
    trainset = custom_dataset(images[:-300],labels[:-300],annotations[:-300])
    train_loader = DataLoader(trainset, batch_size=batch_size, \
                                        shuffle=True)
    valset = custom_dataset(images[-300:],labels[-300:],annotations[-300:])
    val_loader = DataLoader(valset, batch_size=batch_size)
    config = DTrOCRConfig(
        # attn_implementation='flash_attention_2'
    )
    tk=MBart50TokenizerFast.from_pretrained(config.mbart_hf_model)
    for t in train_loader:
        break
    print([tk.decode(i) for i in t['input_ids'][0]])
    Image.fromarray((t['pixel_values'][0].permute(1,2,0).numpy()*255).astype(np.uint8))