#TODO optimize, use pytorch's numpy
class Memory:

    def __init__(self):
        self.actions = None
        self.states = None
        self.logprobs = None
        self.rewards = None
        self.is_terminals = None
        self.clear()

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.actions)
