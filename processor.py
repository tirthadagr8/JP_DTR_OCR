from transformers import MBart50TokenizerFast,ViTImageProcessor
from data import *
from config import *
from PIL import Image
from typing import List, Union

class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        self.vit_processor = ViTImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={
                "height": config.image_size[0],
                'width': config.image_size[1]
            },
        )
        self.tokeniser = MBart50TokenizerFast.from_pretrained(
            config.mbart_hf_model,
            add_bos_token=add_bos_token,
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )
#         self.tokeniser.pad_token = self.tokeniser.bos_token
        self.tokeniser.add_eos_token = add_eos_token
        self.tokeniser.add_bos_token= add_bos_token
        # Bind a new method to gpt2_tokeniser
#         self.tokeniser.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(
#             self.tokeniser
#         )

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        text_inputs = self.tokeniser(
            texts, return_tensors='pt', padding=padding, truncation=True,# max_length=self.max_length
        ) if texts is not None else None

        image_inputs = self.vit_processor(
            images,return_tensors='pt'
        ) if images is not None else None

        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"] if images is not None else None,
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None
        )