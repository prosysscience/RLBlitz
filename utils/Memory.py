#TODO optimize, use pytorch's numpy
class Memory:

    def __init__(self):
        self.actions = None
        self.states = None
        self.values = None
        self.logprobs = None
        self.rewards = None
        self.is_terminals = None
        self.entropy = 0
        self.clear()

    def clear(self):
        self.actions = []
        self.states = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.entropy = 0

    def __len__(self):
        return len(self.actions)
