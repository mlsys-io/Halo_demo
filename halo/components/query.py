import uuid
import time

class Query:
    def __init__(self, id, prompt, priority=0):
        self.id = id if id is not None else uuid.uuid4()
        self.prompt = prompt
        self.prompt_len = len(prompt) if prompt is not None else 0
        self.status = "pending"
        self.priority = priority

        self.prefix_len = 0
        self.input_ids = None
        self.generated_tokens = 0
        self.decoded_tokens = ""
        self.new_tokens = 0
        self.op_output = {}
        self.step = 0
        self.cache = None
        
        # benchmark
        self.create_time = time.perf_counter()
        self.benchmark = {}
   