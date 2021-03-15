import os

# from . import __file__

# from resources import _get_resource


# def main():
#     print(_get_resource('test'))

# main()

base = os.path.dirname(os.path.realpath(__file__))
print(base)

file = '/home/kanebako/.local/share/virtualenvs/AI-Feynman-Rp8PuYnV/lib/python3.7/site-packages/aifeynman-2.0.7-py3.7-linux-x86_64.egg/aifeynman/14ops.txt'
with open(file, 'r') as f:
    print(f)


