from agents.a2c import A2C

if __name__ == '__main__':
    a2c = A2C()
    for _ in range(300):
        a2c.train()
    a2c.save_agent_checkpoint()
    del a2c
    a2cReload = A2C.load_agent_checkpoint()
    for _ in range(300):
        a2cReload.train()
    #a2c.render()