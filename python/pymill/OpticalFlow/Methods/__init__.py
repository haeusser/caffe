#!/usr/bin/python
# coding: utf-8

import os

methods = []
for file in os.listdir('/home/ilge/dev/pymill/OpticalFlow/Methods'):
    if file == '__init__.py' or not file.endswith('.py'): continue
    name = file.replace('.py','')
    fullName = 'pymill.OpticalFlow.Methods.%s' % name

    __import__(fullName)
    methods.append(name)
