"""Module with a syntax error in the middle."""


def before(x):
    return x + 1


def broken(:
    pass


class After:
    def method(self):
        return 1
