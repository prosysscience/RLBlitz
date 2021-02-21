from agents.a2c import A2C
from agents.ppo import PPO

if __name__ == '__main__':
    ppo = PPO()
    for _ in range(10000):
        ppo.train()
    #a2c.render()