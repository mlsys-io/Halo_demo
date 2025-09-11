import uuid
class Operator:
    def __init__(self, id=None, prompt=None, model_config=None, keep_cache=False):
        self.id = id if id is not None else uuid.uuid4()
        self.input_ops = []
        self.output_nodes = []
        self.prompt = prompt
        self.model_config = model_config
        self.max_distance = None
        self.keep_cache = keep_cache
        self.benchmark = Benchmark()
        
        #for data parallelism
        self.data_parallel = False
        self.is_duplicate = False
        self.main_node = None
        self.duplicate_info = None

class Benchmark:
    def __init__(self):
        self.init_time = 0
        self.prefill_time = 0
        self.generate_time = 0
        
    def total_time(self):
        return self.init_time + self.prefill_time + self.generate_time
    
    def update(self, dict):
        self.init_time += dict.get('init_time', 0.0)
        self.prefill_time += dict.get('prefill_time', 0.0)
        self.generate_time += dict.get('generate_time', 0.0)

    def __str__(self):
        return f"Init time: {self.init_time}, Prefill time: {self.prefill_time}, Generate time: {self.generate_time}, Total time: {self.total_time()}"