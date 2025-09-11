class ModelConfig:
    def __init__(
        self, 
        model_name, 
        system_prompt=None, 
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=256,
        max_batch_size=8,
        dtype='bfloat16',
        use_chat_template=True,
        quantization=None,
        lora_config=None,
        max_model_len=None,
        min_tokens=0,
        ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.common_message = ''
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.dtype = dtype  
        self.use_chat_template = use_chat_template
        self.quantization = quantization
        self.lora_config = lora_config
        self.max_model_len = max_model_len
        self.min_tokens = min_tokens