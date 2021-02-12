from agents.a2c import A2C
from agents.ppo import PPO

if __name__ == '__main__':
    a2c = A2C()
    for _ in range(1000):
        a2c.train()
    #a2c.render()