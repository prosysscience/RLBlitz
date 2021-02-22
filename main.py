from agents.a2c import A2C
from agents.ppo import PPO

if __name__ == '__main__':
    algo = A2C()
    for _ in range(1000):
        algo.train()
    #algo.render()