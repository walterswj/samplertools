# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 07:52:59 2022

@author: w_wal
"""
import samplertools
from samplertools import SamplerDatabase
import numpy as np

#keff=np.genfromtxt('nodiversion/keff.out')

#samplertools.gen_h5_from_csv('nodiversion*/msdr.samplerfiles/*.f71.csv','nodiv_1000.h5')
samplertools.gen_h5_from_csv('diversion*/msdr.samplerfiles/*.f71.csv','div_1000.h5')


sampler=SamplerDatabase('div_1000.h5')

print(sampler.get_mean(942390,1))
print(sampler.get_unc(942390,1))

