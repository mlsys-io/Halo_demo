from .operator import Operator
class ExecuteInfo:
    def __init__(self, op: Operator, query_ids, prompts):
        self.op = op
        self.query_ids = query_ids
        self.prompts = prompts