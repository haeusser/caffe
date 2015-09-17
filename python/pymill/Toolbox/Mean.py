#!/usr/bin/python

class Mean:
    def __init__(self):
        self._sum = float(0)
        self._count = float(0)

    def consider(self, x):
        self._sum += x
        self._count += 1

    def mean(self):
        return self._sum / float(self._count)

def dictionaryListMean(list):
    results = {}
    for entry in list:
        for field in entry:
            if field not in results:
                results[field] = entry[field]
            else:
                results[field] += entry[field]


    for field in results:
        results[field] /= float(len(list))

    return results