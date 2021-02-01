from agents.a2c import A2C

if __name__ == '__main__':
    a2c = A2C()
    a2c.act()
    a2c.update()
    a2c.act()
