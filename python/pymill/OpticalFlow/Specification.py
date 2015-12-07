#!/usr/bin/python
# coding: utf-8

import re

class Specification:
    def __init__(self,buf=None):
        self._direction = ''
        self._params = {}
        self._inputs = []
        self._name = ''

        if buf:
            buf=self.parse(buf)

            if buf != '':
                raise Exception('don\'t know how to parse: %s' % buf)
                raise Exception('don\'t know how to parse: %s' % buf)

    def direction(self): return self._direction
    def params(self): return self._params
    def inputs(self): return self._inputs
    def name(self): return self._name

    def parseIdentifier(self, buf):
        match = re.compile('^((_|\.|[a-z]|[A-Z])(_|\.|[a-z]|[A-Z]|[0-9])*)').match(buf)

        if not match:
            raise Exception('failed to parse identifier at: %s' % buf)

        identifier = match.group(1)
        return (buf[len(identifier):], identifier)

    def parseValue(self,buf):

        match = re.compile('^((_|\.|[a-z]|[A-Z]|[0-9])+)').match(buf)
        if not match:
            raise Exception('failed to parse value at: %s' % buf)

        value = match.group(1)
        return (buf[len(value):],value)

    def parseInput(self, buf):
        try:
            input = Specification()
            buf = input.parse(buf)
        except Exception as e:
            raise Exception('failed to parse input at: %s' % buf)

        self._inputs.append(input)

        return buf

    def parseParameters(self, buf):
        buf, identifier = self.parseIdentifier(buf)

        if buf[0] != '=':
            raise Exception('expected = at: %s' % buf)
        buf = buf[1:]

        buf, value = self.parseValue(buf)

        self._params[identifier] = value

        while True:
            if len(buf) == 0 or buf[0] != ',':
                return buf
            buf = buf[1:]

            buf, identifier = self.parseIdentifier(buf)

            if buf[0] != '=':
                raise Exception('expected = at: %s' % buf)
            buf = buf[1:]

            buf, value = self.parseValue(buf)

            self._params[identifier] = value

    def parse(self, buf):
        if buf.startswith('+'):
            self._direction = '+'
            buf = buf[1:]
        elif buf.startswith('-'):
            self._direction = '-'
            buf = buf[1:]

        buf, self._name = self.parseIdentifier(buf)
        if buf.startswith('['):
            buf = buf[1:]
            while buf[0] != ']':
                buf = self.parseInput(buf)
                if buf[0] == ',':
                    buf = buf[1:]

            if buf[0] != ']':
                raise Exception('expected ] at: %s' % buf)
            buf = buf[1:]

        if buf.startswith('@'):
            buf = buf[1:]
            buf = self.parseParameters(buf)

        return buf

    def __repr__(self):
        buf=self._direction+self._name

        for input in self._inputs:
            buf += '['
            buf += str(input)
            buf += ']'

        if len(self._params):
            buf += '@'

            first = True
            for key in sorted(self._params.iterkeys()):
                if not first: buf += ','
                buf += key
                buf += '='
                buf += self._params[key]
                first = False

        return buf

    @staticmethod
    def expand(buf):
        return [Specification(part) for part in buf.split(':')]
