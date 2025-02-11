import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import CLIPProcessor, CLIPModel
from janus.models import MultiModalityCausalLM, VLChatProcessor

def set_dtype_device(model, precision=16):
    dtype = (torch.bfloat16 if torch.cuda.is_available() else torch.float16) if precision==16 else (torch.bfloat32 if torch.cuda.is_available() else torch.float32)
    cuda_device = 'cuda' if torch.cuda.is_available() else 'mps'
    if torch.cuda.is_available():
        model = model.to(dtype).cuda()
    else:
        torch.set_default_device("mps")
        model = model.to(dtype)
    return model, dtype, cuda_device


class Model_Utils:
    def __init__(self):
        pass
    
    def prepare_inputs(self):
        raise NotImplementedError

    def generate_outputs(self):
        raise NotImplementedError



class Clip_Utils(Model_Utils):
    def __init__(self):
        super().__init__()

    def init_Clip(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def prepare_inputs(self, question_lst, image):
        inputs = self.processor(text=question_lst, images=image, return_tensors="pt", padding=True)
        return inputs
        

class Janus_Utils(Model_Utils):
    def __init__(self):
        super().__init__()

    
    def init_Janus(self, num_params="1B"):

        model_path = f"deepseek-ai/Janus-Pro-{num_params}"
        config = AutoConfig.from_pretrained(model_path)
        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                                    language_config=language_config,
                                                    trust_remote_code=True,
                                                    ignore_mismatched_sizes=True)
        self.vl_gpt, self.dtype, self.cuda_device = set_dtype_device(self.vl_gpt)
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        return self.vl_gpt, self.tokenizer
    
    def prepare_inputs(self, question, image):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        pil_images = [Image.fromarray(image)]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.cuda_device, dtype=self.dtype)

        return prepare_inputs
    
    def generate_inputs_embeddings(self, prepare_inputs):
        return self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    def generate_outputs(self, inputs_embeds, prepare_inputs, temperature, top_p, with_attn=False):
        
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False if temperature == 0 else True,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
        )

        return outputs

    
    

    

    

    