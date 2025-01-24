import torch
import torch.nn
import os
import time
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DTrOCRLMHeadModel
from config import DTrOCRConfig
from dataset import *
from datetime import datetime


config=DTrOCRConfig()
torch.set_float32_matmul_precision('high')
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DTrOCRLMHeadModel(config)
if os.path.exists('./dtrocr_2024-11-07.pth'):
    model.load_state_dict(torch.load('./dtrocr_2024-11-07.pth',weights_only=True,map_location=device))
    print('Loaded weights')
# model = torch.compile(model)
model.to('cpu')
model.eval()

with torch.no_grad():
    test_processor = DTrOCRProcessor(DTrOCRConfig())
    image_folder = "C:/Users/tirth/Desktop/DTR_OCR/"
    ans=[]
    image_files=sorted(glob.glob(image_folder+'*jpg'))
    print('Found Images:',len(image_files))
    for image_file in tqdm(image_files):
        image = Image.open(image_file).convert('RGB')

        inputs = test_processor(
            images=image, 
            texts='',#test_processor.tokeniser.bos_token,
            return_tensors='pt'
        )
        print(test_processor.tokeniser.batch_decode(inputs.input_ids))
        inputs.input_ids=inputs.input_ids[:,:1]
        model_output = model.generate(
            inputs, 
            test_processor, 
            num_beams=5,
            # use_cache=True
        )

        predicted_text = test_processor.tokeniser.decode(model_output[0])
        print(predicted_text)
        ans.append(predicted_text+'\n')

    #     print(predicted_text)
    #     plt.figure(figsize=(10, 5))
    # #     plt.title(predicted_text, fontsize=24)
    #     plt.imshow(np.array(image, dtype=np.uint8))
    #     plt.xticks([]), plt.yticks([])
    #     plt.show()
    with open('ans.txt','w',encoding='utf-8') as f:
        f.writelines(ans)