from collections import OrderedDict, Counter

import sys
from caffe.proto import caffe_pb2
from google import protobuf
import six
from termcolor import colored

def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    param_names = [s for s in dir(layer) if s.endswith('_param')]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # strip the final '_param' or 'Parameter'
    param_names = [s[:-len('_param')] for s in param_names]
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]
    return dict(zip(param_type_names, param_names))


def to_proto(*tops):
    """Generate a NetParameter that contains all layers needed to compute
    all arguments."""


    layers = OrderedDict()
    autonames = Counter()
    for top in tops:
        top.layer._to_proto(layers, {}, autonames)
    net = caffe_pb2.NetParameter()
    net.layer.extend(layers.values())
    return net


def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly."""

    if isinstance(val, list) or isinstance(val, tuple):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in six.iteritems(item):
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in six.iteritems(val):
            assign_proto(getattr(proto, name), k, v)
    else:
        #try:
            #getattr(proto, name)
        #except:
            #return
        setattr(proto, name, val)


class Blob(object):
    """A Top specifies a single output blob (which could be one of several
    produced by a layer.)"""

    def __init__(self, net):
        self._net = net
        self._name = None
        self._output = False
        self._inputRefCount = 0
        self._outputRefCount = 0
        self._isSibling = False
        self._master = None

    def name(self): return self._name
    def net(self): return self._net
    def output(self): return self._output
    def inputRefCount(self): return self._inputRefCount
    def outputRefCount(self): return self._outputRefCount

    def makeSibling(self, master):
        if master == self: return

        self._isSibling = True
        self._master = master

    def isSibling(self): return self._isSibling

    def takeMasterName(self):
        self._name = self._master.name()

    def setName(self, name):
        self._name = name
        if self._isSibling:
            self._master.setName(name)
        else:
            self._net.blobs()[name] = self

    def incInputRef(self):
        self._inputRefCount += 1

    def incOutputRef(self):
        self._outputRefCount += 1

    def used(self):
        return self.inputRefCount() or self.outputRefCount()

    def enableOutput(self):
        self._output = True
        return self

class Layer(object):
    """A Function specifies a layer, its parameters, and its inputs (which
    are Tops from other layers)."""

    def __init__(self, typeName, inputs, params):
        self._typeName = typeName
        self._net = inputs[0]
        self._inputs = inputs[1:]
        if len(self._inputs) and (isinstance(self._inputs[0],tuple) or isinstance(self._inputs[0],list)):
            self._inputs = self._inputs[0]

        self._params = params
        self._in_place = self._params.get('in_place', False)
        if 'in_place' in self._params:
            del self._params['in_place']

        if 'name' not in self._params:
            self._params['name'] = self._net.finalizeName(self._net.newLayerName(typeName))

        if 'nout' in self._params:
            ntop = self._params['nout']
            del self._params['nout']
        else:
            ntop = 0

        if self._in_place:
            self._outputs = self._inputs
            ntop = len(self._outputs)
        else:
            self._outputs = []
            for i in range(ntop):
                self._outputs.append(self._net.newBlob())

        for input  in self._inputs:  input.incInputRef()
        for output in self._outputs: output.incOutputRef()

        self._net.addLayer(self)

    def inputs(self): return self._inputs
    def outputs(self): return self._outputs

    def toProto(self):
        layer = caffe_pb2.LayerParameter()
        layer.type = self._typeName
        for blob in self._inputs:  layer.bottom.append(blob.name())
        for blob in self._outputs: layer.top.append(blob.name())

        for k, v in six.iteritems(self._params):
            # special case to handle generic *params
            if k.endswith('param'):
                assign_proto(layer, k, v)
            else:
                try:
                    assign_proto(getattr(layer,
                        _param_names[self._typeName] + '_param'), k, v)
                except (AttributeError, KeyError):
                    assign_proto(layer, k, v)

        return layer


class DataStruct(object):
    def __init__(self, prefix = ''):
        super(DataStruct, self).__setattr__('_members', OrderedDict())
        super(DataStruct, self).__setattr__('_suffixes', OrderedDict())
        super(DataStruct, self).__setattr__('_prefix', prefix)

    def add(self, name, suffix=None):
        if name in self._members:
            raise Exception('stage %s already exists' % name)

        if suffix is None:
            suffix = '_' + name

        member = DataStruct()
        self._members[name] = member
        self._suffixes[name] = suffix

        return member

    def dict(self):
        dict = {}
        for name, member in self._members.iteritems():
            if isinstance(member, Blob):
                dict[self._prefix + name] = member
            elif isinstance(member, DataStruct):
                subdict = member.dict()
                for subName, subBlob in subdict.iteritems():
                    dict[self._prefix + name + '_' + subName] = subBlob
        return dict

    def blobs(self):
        return self.dict().values()

    def copyNamesTo(self, other):
        if isinstance(other, Network):
            other = other.namedBlobs()

        dict = self.dict()
        for key, value in dict.iteritems():
            other[key] = value

    def __getattr__(self, item):
        item = str(item)

        for name, member in self._members.iteritems():
            if item == name:
                return member

        return self.add(item)

    def __setattr__(self, index, value):
        index=str(index)

        if index in self._members:
            value.makeSibling(self._members[index])
        else:
             self._members[index] = value

        if not isinstance(value, (Blob, DataStruct)):
            raise Exception('tried to set a member of DataStruct that is not of type Blob or DataStruct')

        self._members[index] = value
        self._suffixes[index] = '_' + index

    def makeSibling(self, other):
        for name, member in other._members.iteritems():
            self._members[name].makeSibling(member)

    def __iter__(self):
        return iter(self.dict().keys())

    def iteritems(self):
        return self.dict().iteritems()

    def __getitem__(self, index):
        return self.__getattr__(index)

    def __setitem__(self, index, value):
        return self.__setattr__(index, value)

NamedBlobs = DataStruct

class Network(object):
    """A NetSpec contains a set of Tops (assigned directly as attributes).
    Calling NetSpec.to_proto generates a NetParameter containing all of the
    layers needed to produce all of the assigned Tops, using the assigned
    names."""

    def __init__(self, prefix=''):
        self._members = DataStruct()
        self._layers = []
        self._blobs = []
        self._prefix = prefix
        self._counts = {}
        self.layers = {}

    def namedBlobs(self):
        return self._members

    def blobs(self):
        return self._members

    def addLayer(self, layer):
        self._layers.append(layer)

    def newBlob(self):
        blob = Blob(self)
        self._blobs.append(blob)
        return blob

    def finalizeName(self, name):
        if self._prefix != '':
            return self._prefix + '_' + name
        return name

    def newLayerName(self, type):
        i = 1
        if type in self._counts:
            self._counts[type] += 1
            i = self._counts[type]
        else:
            self._counts[type] = i

        return '%s%d' % (type, i)

    def toProto(self):
        i = 0
        for blob in self._blobs:
            if blob.isSibling(): continue
            if blob.used() and blob.name() is None:
                if blob in self._members.dict().values():
                    for member in self._members.dict():
                        if self._members.dict()[member] == blob:
                            blob.setName(member)
                            break
                else:
                    blob.setName(self.finalizeName('blob%d' % i))
                i += 1

        for blob in self._blobs:
            if blob.isSibling():
                blob.takeMasterName()

        for blob in self._blobs:
            if blob.used() and blob.inputRefCount() == 0 and not blob.output() and not blob.isSibling():
                sys.stderr.write(colored('silencing blob %s\n' % blob.name(),'red',attrs=['bold']))
                Layers.Silence(self, blob)

        protoLayers = []
        for layer in self._layers:
            protoLayers.append(layer.toProto())

        net = caffe_pb2.NetParameter()
        net.layer.extend(protoLayers)

        return net


class LayerCreator(object):
    """A Layers object is a pseudo-module which generates functions that specify
    layers; e.g., Layers().Convolution(bottom, kernel_size=3) will produce a Top
    specifying a 3x3 convolution applied to bottom."""

    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            layer = Layer(name, args, kwargs)
            outputs = layer.outputs()
            if len(outputs) == 0:
                return layer
            elif len(outputs) == 1:
                return outputs[0]
            else:
                return outputs
        return layer_fn


class ParameterCreator(object):
    """A Parameters object is a pseudo-module which generates constants used
    in layer parameters; e.g., Parameters().Pooling.MAX is the value used
    to specify max pooling."""

    def __getattr__(self, name):
       class Param:
            def __getattr__(self, param_name):
                return getattr(getattr(caffe_pb2, name + 'Parameter'), param_name)
       return Param()


_param_names = param_name_dict()
_param_names['Deconvolution'] = 'convolution'
_param_names['Correlation1D'] = 'correlation'
_param_names['NegReLU'] = 'relu'

Layers = LayerCreator()
Params = ParameterCreator()



# class NamedBlobs(object):
#     def __init__(self):
#         super(NamedBlobs, self).__setattr__('members', OrderedDict())
#
#     def dict(self): return self.members
#
#     def blobs(self): return self.members.values()
#
#     def __setattr__(self, key, value):
#         if isinstance(key, int):
#             key = '_%d' % key
#
#         if key in self.members:
#             value.makeSibling(self.members[key])
#         else:
#             self.members[key] = value
#
#     def __getattr__(self, item):
#         if isinstance(item, int):
#             item = '_%s' % item
#
#         return self.members[item]
#
#     def copyNamesTo(self, other):
#         if isinstance(other, Network):
#             other = other.namedBlobs()
#
#         for key, value in self.members.iteritems():
#             setattr(other, key, value)
#
#     def __iter__(self):
#         return iter(self.members.keys())
#
#     def iteritems(self):
#         return self.dict().iteritems()
#
#     def __getitem__(self, index):
#         return self.__getattr__(index)
#
#     def __setitem__(self, index, value):
#         return self.__setattr__(index, value)
