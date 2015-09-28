from pymill.CNN.Definition import Data as Data

class Dataset:
    def __init__(self, name, rendertype, phase):
        self._name = name
        self._rendertype = rendertype
        self._phase = phase

    def name(self): return self._name
    def rendertype(self): return self._rendertype

class Sintel(Dataset):
    def __init__(self, rendertype, phase):
        Dataset.__init__(self, 'sintel', rendertype, phase)

    def width(self): return 1024
    def height(self): return 436
    def meanColors(self):
        if self._rendertype == 'CLEAN': return (76.4783107737, 69.4660111681, 58.0279756163)
        else:                           return (91.2236713645, 82.6859238723, 69.5627393708)

    def flowLayer(self, net, **kwargs):
        kwargs['setting'] = 'OPTICAL_FLOW_SINGLE'
        kwargs['phase'] = self._phase
        return Data.Sintel(**kwargs)

    def dispLayer(self, net, **kwargs):
        kwargs['setting'] = 'DISPARITY_SINGLE'
        kwargs['phase'] = self._phase
        return Data.Sintel(net, **kwargs)

def get(name=None, rendertype=None, phase=None):
    if name == 'sintel.train.clean': return Sintel('CLEAN', phase)
    if name == 'sintel.train.final': return Sintel('FINAL', phase)
    if name == 'sintel': return Sintel(rendertype, phase)
    else:
        raise Exception('unknown dataset: %s' % name)

