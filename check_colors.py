import os

files = os.listdir('/home/ngs/pillproject/augmented/Color/RED')
for file in files:
    file = file.str[2:]
    