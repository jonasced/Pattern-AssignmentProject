#!/usr/bin/python3

import numpy as np
from DrawCharacter import *

ch1 = DrawCharacter()
ch1.run()

pts = ch1.get_xybpoints()

name = input("What do you want to save the file as?")

np.save(str(name)+".npy",pts)
"""
with open(str(name)+".d","w") as file:
    for row in range(pts.shape[0]):  # prints each row on a separate line
        file.write(str(pts[row]))
        file.write("\n")
        print("New row")
"""
