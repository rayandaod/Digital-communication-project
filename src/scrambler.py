"""
Credits: Prandoni P. - Signal processing for Communication course at EPFL
"""


class Scrambler:
    def __init__(self):
        pass

    buffer = [0] * 23
    ix = 0

    def scramble(self, bit):
        out = bit ^ self.buffer[(self.ix + 17) % 23] ^ self.buffer[(self.ix + 22) % 23]
        self.buffer[self.ix] = out
        self.ix = (self.ix + 1) % 23
        return out


class Descrambler:
    def __init__(self):
        pass

    buffer = [0] * 23
    ix = 0

    def descramble(self, bit):
        out = bit ^ self.buffer[(self.ix + 17) % 23] ^ self.buffer[(self.ix + 22) % 23]
        self.buffer[self.ix] = bit
        self.ix = (self.ix + 1) % 23
        return out
