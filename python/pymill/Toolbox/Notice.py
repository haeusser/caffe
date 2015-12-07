#!/usr/bin/python

from termcolor import colored

def notice(str, type=None):
    if type == 'run':
        print colored(str, 'green')
    elif type == 'skip':
        print colored(str, 'blue', attrs={'bold': 1})
    elif type == 'missing':
        print colored(str, 'red', attrs={'bold': 1})
    elif type == 'del':
        print colored(str, 'yellow', attrs={'bold': 1})
    elif type == 'warning':
        print colored(str, 'yellow', attrs={'bold': 1})
    elif type == 'passed':
        print colored(str, 'green', attrs={'bold': 1})
    elif type == 'failed':
        print colored(str, 'red', attrs={'bold': 1})
    elif type == 'remove':
        print colored(str, 'red', attrs={'bold': 1})
    elif type == 'notice':
        print colored(str, 'cyan', attrs={'bold': 1})
    else:
        print str

def noticeVerbose(str, type=None):
    if not verbose: return
    notice(str,type)

verbose = False
