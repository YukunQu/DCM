# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:27:47 2021

@author: qyk
"""
import pandas as pd 

subinfo = r'C:\Myfile\File\工作\PhD\Development cognitive map\exp\task\dcm_day1\subinfo/subinfo.xlsx'
data = pd.read_excel(subinfo)

name = data['姓名']
names = data['姓名']

mapset = data['mapSet']
mapsets = data['mapSet']

day1 = data['Day1']
day1s = data['Day1']
day2 = data['Day2']
day2s = data['Day2']

for i in range(9):
    names = names.append(name)
    mapsets = mapsets.append(mapset)
    day1s = day1s.append(day1)
    day2s = day2s.append(day2)

participant = range(1,201)
age = []
for i in range(9,19):
    for repeat in range(20):
        age.append(i)

data = pd.DataFrame({'participant':participant,'姓名':names,'mapset':mapsets,
                     'day1':day1s,'day2':day2s,'age':age})
data.to_excel(subinfo,index=False)