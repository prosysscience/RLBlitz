import torch


class Memory:

    def __init__(self, config, state_dim, device):
        self.config = config
        self.state_dim = state_dim
        self.actions = None
        self.states = None
        self.values = None
        self.logprobs = None
        self.rewards = None
        self.is_terminals = None
        self.entropy = 0
        self.device = device
        self.clear()

    def clear(self):
        self.actions = torch.empty((self.config['num_steps'], self.config['num_worker']), dtype=torch.int, device=self.device)
        self.states = torch.empty((self.config['num_steps'], self.config['num_worker'], self.state_dim), dtype=torch.float, device=self.device)
        self.values = torch.empty((self.config['num_steps'], self.config['num_worker'], 1), dtype=torch.float, device=self.device)
        self.logprobs = torch.empty((self.config['num_steps'], self.config['num_worker']), dtype=torch.float, device=self.device)
        self.rewards = torch.empty((self.config['num_steps'], self.config['num_worker']), dtype=torch.float, device=self.device)
        self.is_terminals = torch.empty((self.config['num_steps'], self.config['num_worker']), dtype=torch.bool, device=self.device)
        self.entropy = 0

    def __len__(self):
        return self.config['num_steps']
