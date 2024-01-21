'''
Modified for Candidate-Reranking CIR stage II
https://github.com/salesforce/BLIP/blob/main/models/blip_nlvr.py
'''
import os
from med import BertConfig
from nlvr_encoder import BertModel
from vit import interpolate_pos_embed
from blip import create_vit, init_tokenizer, is_url

from timm.models.hub import download_cached_file

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

class BLIP_NLVR(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 ):
        """
        Modified BLIP_NLVR model for CIR

        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        try:
            self.tokenizer = init_tokenizer()  
        except:
            import pickle
            # in case of huggingface API goes offline (it happens)
            print(f"Something wrong with huggingface API, trying to load a local backup pickle file on self.tokenizer...")
            self.tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))

        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False) 
                    
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size * 2, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  

    
    def img_embed(self, image, train=True, atts=False):
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        if atts:
            return image_embeds, image_atts
        else:
            return image_embeds

    def img_txt_fusion(self, r_image_embeds, t_image_embeds, text, train=True):
        r_image_embeds = r_image_embeds.last_hidden_state
        device = r_image_embeds.device
        t_image_atts = torch.ones(t_image_embeds.size()[:-1],dtype=torch.long).to(device)        

        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        text.input_ids[:,0] = self.tokenizer.enc_token_id        

        images_in_batch = r_image_embeds.shape[0]
        text_input_ids_s, text_attn_mask_s = text.input_ids.shape, text.attention_mask.shape

        # run for-loop for each row (i-th) and collect to form B x B
        hidden_state = torch.empty((0, 768 * 2)).to(device, non_blocking=True) # at the end, B**2, 768
        for i in range(images_in_batch):
            image1_embeds = t_image_embeds # B x dim
            output_i = self.text_encoder(text.input_ids[i].unsqueeze(0).expand(*text_input_ids_s), # B x dim
                                   attention_mask = text.attention_mask[i].unsqueeze(0).expand(*text_attn_mask_s), # B x dim

                                   z_t = r_image_embeds[i].unsqueeze(0).expand(*r_image_embeds.shape), # B x L x dim
                                   z_t_attention_mask = None, # B x dim

                                   encoder_hidden_states = [image1_embeds,image1_embeds],
                                   encoder_attention_mask = [t_image_atts,
                                                             t_image_atts],        
                                   return_dict = True,
                                  )  

            hidden_state_i = output_i

            hidden_state = torch.vstack((hidden_state, hidden_state_i))
        
        prediction = self.cls_head(hidden_state)
        prediction = prediction.view(images_in_batch, images_in_batch, 2) # B x B x 2

        return prediction[:,:,0] # B x B
    
    def img_txt_fusion_val(self, r_image_embeds, t_image_embeds, text):
        '''
        Same function as img_txt_fusion, but accepts batch_size as 1, and K candidates.
        Returns 1 x K  prediction
        '''
        r_image_embeds = r_image_embeds.last_hidden_state
        device = r_image_embeds.device
        assert r_image_embeds.shape[0] == 1 # assume batch_size 1
        K = t_image_embeds.shape[0]

        t_image_atts = torch.ones(t_image_embeds.size()[:-1],dtype=torch.long).to(device)        

        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        text.input_ids[:,0] = self.tokenizer.enc_token_id        

        text_input_ids_s, text_attn_mask_s = text.input_ids.shape, text.attention_mask.shape

        image0_embeds = r_image_embeds.expand(K, *r_image_embeds.shape[1:]) # K, ...
        image1_embeds = t_image_embeds # K x dim
        output_i = self.text_encoder(text.input_ids.expand(K, *text_input_ids_s[1:]), 
                                attention_mask = text.attention_mask.expand(K, *text_attn_mask_s[1:]), 

                                z_t = image0_embeds,
                                z_t_attention_mask = None,

                                encoder_hidden_states = [image1_embeds,image1_embeds],
                                encoder_attention_mask = [t_image_atts,
                                                            t_image_atts],        
                                return_dict = True,
                                )  

        hidden_state_i = output_i

        prediction = self.cls_head(hidden_state_i) # K x 2
        
        return prediction[:,0] # K
    

def blip_stage2(pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model  

        
def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key] 

        if 'attention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'attention.output.dense.' in key:
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]  
        
        if 'output.LayerNorm' in key and 'attention' in key: # text_encoder.encoder.layer.0.crossattention.output.LayerNormA/B.bias
            new_key0 = key.replace('LayerNorm','LayerNormA')
            new_key1 = key.replace('LayerNorm','LayerNormB')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]

    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
            