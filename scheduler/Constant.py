class Constant:

    def __init__(self, intial='1e-4'):
        self.value = intial

    def get_value(self):
        return self.value

    def __str__(self):
        return 'Constant value {}'.format(self.value)
