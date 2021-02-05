from agents.a2c import A2C

if __name__ == '__main__':
    a2c = A2C()
    for _ in range(5000):
        a2c.train()
    a2c.save_model()
    #a2c.render()