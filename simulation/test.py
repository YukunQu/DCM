# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:32:48 2021

@author: qyk
"""

import numpy as np
from simulation import utils
import matplotlib.pyplot as plt


def testSimulation():
    angles = np.linspace(0,361,360)
    activation = utils.genSimulateData(angles,0)
    
    plt.plot(np.deg2rad(angles),activation)
    plt.xtick()
    plt.show()