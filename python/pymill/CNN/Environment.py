#!/usr/bin/python

import os
import sys
import matplotlib.pyplot as plt
from pymill import Toolbox as tb
import re
from Files import iterFiles
from Files import logFiles
from Log import Log
from pymill import Config
from string import Template


def caffeBin():
    bin = os.environ['CAFFE_BIN']
    if not bin:
        raise Exception(
            'You do not have your environment variable CAFFE_BIN set. Example: export CAFFE_BIN=\'/home/ilge/dev/hackathon-caffe/build/tools/caffe.bin\'')

    if not os.path.isfile(bin) or not os.access(bin, os.X_OK):
        raise Exception('You CAFFE_BIN environment variable does not point to a valid binary')

    return bin


class BinaryBackend:
    def __init__(self, gpus='0', quiet=False, silent=False):
        self._gpus = gpus
        self._silent = silent
        self._quiet = quiet

    def _callBin(self, cmd):
        cmd = 'GLOG_logtostderr=%d %s %s' % (not self._quiet, caffeBin(), cmd)
        if not self._silent:
            tb.notice('running "%s"' % cmd, 'run')
        tb.system(cmd)

    def train(self, solverFilename, logFile, weights=None):
        if weights is not None:
            weightOption = '-weights %s' % weights
        else:
            weightOption = ''

        self._callBin(Template('train -sighup_effect snapshot -solver $solverFilename $weightOption -gpu $gpu 2>&1 | tee -a $logFile').substitute({
            'solverFilename': solverFilename,
            'weightOption': weightOption,
            'gpu': self._gpus,
            'logFile': logFile
        }))

    def resume(self, solverFilename, solverstateFilename, logFile):
        self._callBin(Template(
            'train -sighup_effect snapshot -solver $solverFilename -snapshot $solverstateFilename -gpu $gpu 2>&1 | tee -a $logFile').substitute(
            {
                'solverFilename': solverFilename,
                'solverstateFilename': solverstateFilename,
                'gpu': self._gpus,
                'logFile': logFile
            }))

    def run(self, solverFilename):
        self._callBin(Template('train -solver $solverFilename 2>&1').substitute({
            'solverFilename': solverFilename,
            'gpu': self._gpus
        }))

    def test(self, caffemodelFilename, protoFilename, iterations):
        self._callBin(Template(
            'test -weights $caffemodelFilename -model $protoFilename -iterations $iterations -gpu $gpu 2>&1').substitute(
            {
                'caffemodelFilename': caffemodelFilename,
                'protoFilename': protoFilename,
                'gpu': self._gpus,
                'iterations': iterations
            }))


class PythonBackend(BinaryBackend):
    def __init__(self, gpus='0', quiet=False, silent=False):
        self._gpus = gpus
        self._silent = silent
        self._quiet = quiet
        self.solver = None

    def train(self, solverFilename, logFile, weights=None):
        # hackaround: this backend doesn't log to one file but to a db in a subdirectory

        from pymill.CNN import MillSolver as ms
        self.solver = ms.MillSolver(solver_def=solverFilename, weights=weights, gpus=[int(x) for x in self._gpus.split(',')])
        self.solver.run_solver()

    def resume(self, solverFilename, solverstateFilename, logFile):
        # hackaround: this backend doesn't log to one file but to a db in a subdirectory
        log_dir = os.path.abspath(logFile)
        from pymill.CNN import MillSolver as ms
        self.solver = ms.MillSolver(solver_def=solverFilename, solver_state=solverstateFilename, gpus=[int(x) for x in self._gpus.split(',')])
        self.solver.run_solver()

    def run(self, solverFilename):
        from pymill.CNN import MillSolver as ms
        self.solver = ms.MillSolver(solverFilename=solverFilename, gpus=[int(x) for x in self._gpus.split(',')])
        self.solver.run_solver()

    def get_log_dir_from_filename(self, logFile):
        log_dir = os.path.abspath(logFile)
        log_dir = os.path.join(log_dir[1:log_dir.rfind('/')], 'log')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)


class Environment:
    class Parameters:
        def __init__(self, env):
            self._lines = ''
            self._gpu_arch = 'any'
            self._env = env

        def read(self, filename):
            self._lines = open(filename).readlines()

            for line in self._lines:
                line = line.strip()
                if line.strip() == '': continue
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                if key == 'gpu-arch':
                    self._gpu_arch = value
                elif key == 'name':
                    self._env._name = value
                else:
                    raise Exception('invalid entry in params.txt: %s' % value)

        def gpuArch(self):
            return self._gpu_arch

    def __init__(self, path='.', backend=BinaryBackend(), unattended=False, silent=False):
        self._path = os.path.abspath(path)
        self._backend = backend
        self._unattended = unattended
        self._silent = silent

    def haveTrainDir(self):
        return os.path.isdir(self._trainDir)

    def haveLogDir(self):
        return os.path.isdir(self._logDir)

    def haveJobDir(self):
        return os.path.isdir(self._jobDir)

    def haveScratchDir(self):
        return os.path.isdir(self._scratchDir)

    def haveLogFile(self):
        return os.path.isfile(self._logFile)

    def makeTrainDir(self):
        tb.system('mkdir -p %s' % self._trainDir)
        tb.system('mkdir -p %s' % self._logDir)

    def makeScratchDir(self):
        tb.system('mkdir -p %s' % self._scratchDir)

    def makeJobDir(self):
        tb.system('mkdir -p %s' % self._jobDir)

    def prototxt(self, inFile, outDir, defs={}):
        defs['name'] = self._name

        if not os.path.isfile(inFile):
            raise Exception('input file %s not file' % inFile)

        if inFile.endswith('.prototxt'):
            os.system('cp %s %s' % (inFile, outDir))
            return '%s' % (inFile)
        elif inFile.endswith('.prototmp'):
            prototxt = '%s/%s.prototxt' % (outDir, os.path.basename(inFile).replace('.prototmp', ''))
            if not self._silent: tb.notice('preprocessing %s' % inFile, 'run')
            tb.preprocessFile(inFile, prototxt, defs)
            return prototxt
        elif inFile.endswith('.py'):
            prototxt = '%s/%s.prototxt' % (outDir, os.path.basename(inFile).replace('.py', ''))
            args = ''
            for k, v in defs.iteritems():
                if len(args): args += ' '
                args += '%s=%s' % (k, v)

            if not self._silent:
                if not len(defs):
                    tb.notice('converting %s' % inFile, 'run')
                else:
                    tb.notice('converting %s (%s)' % (inFile, args), 'run')
            if os.system('python %s %s > %s' % (inFile, args, prototxt)) != 0:
                raise Exception('conversion of %s failed' % inFile)
            return prototxt
        else:
            raise Exception('don\'t know how to convert file %s to prototxt' % inFile)

    def makeTrainingPrototxt(self, file, defs={}):
        return self.prototxt(file, self._trainDir, defs)

    def makeScratchPrototxt(self, file, defs={}):
        return self.prototxt(file, self._scratchDir, defs)

    def findProto(self, baseName):
        proto = '%s/%s.py' % (self._path, baseName)
        if not os.path.isfile(proto): proto = '%s/%s.prototmp' % (self._path, baseName)
        if not os.path.isfile(proto): proto = '%s/%s.prototxt' % (self._path, baseName)
        if not os.path.isfile(proto): proto = None
        return proto

    def init(self):
        path = self._path
        self._name = os.path.basename(path)

        self._logDir = self._path + '/training/log'
        self._trainDir = self._path + '/training'
        self._scratchDir = self._path + '/scratch'
        self._jobDir = self._path + '/jobs'

        self._modelFiles = iterFiles('.caffemodel', self._trainDir)
        self._stateFiles = iterFiles('.solverstate', self._trainDir)
        self._logFiles = logFiles(self._logDir)

        self._testProto = self.findProto('test')
        self._trainProto = self.findProto('train')
        self._solverProto = self.findProto('solver')

        self._logFile = self._path + '/training/log.txt'

        self._params = Environment.Parameters(self)
        if os.path.isfile(self._path + '/params.txt'):
            self._params.read(self._path + '/params.txt')

    def notice(self, message, type=None):
        if self._silent:
            return

        if type is None:
            print '%s\n' % message
        else:
            tb.notice(message, type)

    def jobDir(self):
        return self._jobDir

    def params(self):
        return self._params

    def name(self):
        return self._name

    def path(self):
        return self._path

    def modelFiles(self):
        return self._modelFiles

    def stateFiles(self):
        return self._stateFiles

    def logFiles(self):
        return self._logFiles

    def existingData(self, startIter=-1):
        if not self.haveTrainDir():
            return False

        for file in self._modelFiles:
            if file.iteration() > startIter: return True

        for file in self._stateFiles:
            if file.iteration() > startIter: return True

        for file in self._logFiles:
            if file.iteration() > startIter: return True

        return False

    def getModelFile(self, iter=-1):
        if iter == -1:
            iter = self._modelFiles[-1].iteration()
            modelFile = self._modelFiles[-1].filename()
        else:
            for file in self._modelFiles:
                if file.iteration() == iter:
                    modelFile = file.filename()
            if iter == -1:
                raise Exception('no .caffemodel found for iteration %d' % iter)

        return (modelFile, iter)

    def getStateFile(self, iter=-1):
        stateFile = ''
        if iter == -1:
            iter = self._stateFiles[-1].iteration()
            stateFile = self._stateFiles[-1].filename()
        else:
            for file in self._stateFiles:
                if file.iteration() == iter:
                    stateFile = file.filename()
            if iter == -1:
                raise Exception('no .solverstate found for iteration %d' % iter)

        return (stateFile, iter)

    def truncateLog(self, startIter):
        log = Log(self._logFile)
        log.writeUpTo(self._logFile, startIter)

    def clean(self, startIter=-1):
        if self.haveTrainDir():
            if startIter == -1:
                print 'cleaning...'
            else:
                print 'cleaning after iteration %d...' % startIter

            for file in self._modelFiles:
                if file.iteration() > startIter: file.delete(True)

            for file in self._stateFiles:
                if file.iteration() > startIter: file.delete(True)

            for file in self._logFiles:
                if file.iteration() > startIter: file.delete(True)

            if os.path.isfile(self._logFile):
                if startIter != -1:
                    self.truncateLog(startIter)
                    self.notice('truncating training/log.txt after iteration %d' % startIter, 'del')
                else:
                    self.notice('removing training/log.txt', 'del')
                    os.remove(self._logFile)

    def sanitize(self):
        self.notice('removing *.pyc', 'del')
        os.system('rm -f %s/*.pyc' % (self._path, file))

        if self.haveTrainDir():
            self.notice('removing training', 'del')
            os.system('rm -rf %s' % self._trainDir)

        if self.haveScratchDir():
            self.notice('removing scratch', 'del')
            os.system('rm -rf %s' % self._scratchDir)

        if self.haveJobDir():
            self.notice('removing jobs', 'del')
            os.system('rm -rf %s' % self._jobDir)

        for file in os.listdir(self._path):
            if os.path.isdir(file) and file.startswith('output'):
                self.notice('removing %s' % file, 'del')
                os.system('rm -rf %s/%s' % (self._path, file))

    def cleanScratch(self):
        if self.haveScratchDir():
            self.notice('cleaning scratch...', 'del')
            os.system('rm -rf %s' % self._scratchDir)

    def prepareTraining(self):
        self.makeTrainDir()

        if self._testProto != None:
            self.makeTrainingPrototxt(self._testProto)

        self.makeTrainingPrototxt(self._trainProto)
        return self.makeTrainingPrototxt(self._solverProto)

    def train(self, weights=None):
        if self.existingData() and not self._unattended:
            if not tb.queryYesNo('Existing data found. Do you want to delete it and start from scratch?'):
                return

        self.clean()
        solverFilename = self.prepareTraining()

        self.notice('training...')
        os.chdir(self._trainDir)
        if weights != '':
            self._backend.train(solverFilename=solverFilename, logFile=self._logFile, weights=weights)
        else:
            self._backend.train(solverFilename=solverFilename, logFile=self._logFile)

    def resume(self, iter=-1):
        if not len(self._stateFiles):
            raise Exception('no .solverstate files to continue from')

        stateFile, iter = self.getStateFile(iter)

        if self.existingData(iter) and not self._unattended:
            if not tb.queryYesNo(
                            'Existing data beyond iteration %d found. Do you want to delete it and continue?' % iter):
                return

        self.clean(iter)
        solverFilename = self.prepareTraining()

        self.notice('continuing from iteration %d ...' % iter, 'notice')
        os.chdir(self._trainDir)
        self._backend.resume(solverFilename=solverFilename, solverstateFilename=stateFile, logFile=self._logFile)

    def test(self, iter, output=False, definition=None, vars={}, num_iter=-1):
        modelFile, iter = self.getModelFile(iter)

        if output:
            vars['output'] = True
            vars['prefix'] = iter

        self.cleanScratch()
        self.makeScratchDir()

        if definition is None: definition = 'test'
        proto = self.findProto(definition)

        finalProto = self.makeScratchPrototxt(proto, vars)
        solverProto = self.makeScratchPrototxt(self._solverProto, vars)

        self.notice('testing snapshot iteration %d for %d iterations...' % (iter, num_iter), 'notice')
        os.chdir(self._path)
        self._backend.test(caffemodelFilename=modelFile, protoFilename=finalProto, iterations=num_iter)

    def plot(self, select):
        if not self.haveLogFile():
            raise Exception('logfile doesn\'t exist')

        log = Log(self._logFile)
        log.plot(self._name, select)
        plt.show()

    def plotLR(self):
        if not self.haveLogFile():
            raise Exception('logfile doesn\'t exist')

        log = Log(self._logFile)
        log.plotlr(self._name)
        plt.show()

    def view(self, iter):
        raise Exception('under construction')

        self.preprocessFile(self._testPrototmp, self._scratchDir + '/test.prototxt')

        modelFile, iter = self.getModelFile(iter)

        tb.system('weight-viewer %s %s' % (self._scratchDir + '/test.prototxt', modelFile))

    def copy(self, source, target, copySnapshot, iter):
        tb.system('mkdir -p %s' % target)

        for f in os.listdir(source):
            if f == '.': continue
            if f == '..': continue
            if f == 'training':
                if copySnapshot:
                    os.system('mkdir -p %s/training' % target)

                    modelFiles = iterFiles('.caffemodel', '%s/training' % source)
                    stateFiles = iterFiles('.solverstate', '%s/training' % source)

                    if iter != -1:
                        for m in modelFiles:
                            if m.iteration() == iter:
                                tb.system('cp -v %s %s/training' % (m.filename(), target))

                        for s in stateFiles:
                            if s.iteration() == iter:
                                tb.system('cp -v %s %s/training' % (s.filename(), target))
                    else:
                        tb.system('cp -v %s %s/training' % (modelFiles[-1].filename(), target))
                        tb.system('cp -v %s %s/training' % (stateFiles[-1].filename(), target))

                    tb.system('cp %s %s/training/log.txt %s/training' % ('' if self._silent else '-v', source, target))
                continue
            if f == 'scratch': continue
            if f == 'jobs': continue
            if f.endswith('.pyc'): continue
            if os.path.isdir('%s/%s' % (source,f)) and f.startswith('test_'): continue
            if os.path.isdir('%s/%s' % (source,f)) and f.startswith('output_'): continue

            tb.system('cp -r %s %s/%s %s' % ('' if self._silent else '-v', source, f, target))

    def execute(self, file, iter):
        self.makeScratchDir()
        finalProto = self.makeScratchPrototxt(file)

        if iter == -1: iter = 1

        # test requires weighs to be given,
        # therefore we have to use a sovler

        tmpsolver = self._scratchDir + '/run_solver.prototxt'

        f = open(tmpsolver, 'w')
        f.write('train_net: "%s"\n' % finalProto)
        f.write('max_iter: %d\n' % iter)
        f.write('lr_policy: "fixed"\n')
        f.write('snapshot: 0\n')
        f.write('snapshot_after_train: false\n')
        f.close()

        self.notice('running %s for %d iterations ...' % (file, iter), 'notice')
        os.chdir(self._path)
        self._backend.run(tmpsolver)

    def draw(self, prototmp):
        from google.protobuf import text_format
        import caffe, caffe.draw
        from caffe.proto import caffe_pb2

        if not prototmp.endswith('prototxt'):
            self.notice('you need to provide a prototxt file (as of now ...)')
            sys.exit(1)
        else:
            prototxt = prototmp

        outfile = self._scratchDir + '/%s.png' % os.path.basename(prototxt).replace('.prototxt', '')
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt).read(), net)
        self.notice('drawing net to %s' % outfile)
        try:
            caffe.draw.draw_net_to_file(net, outfile, 'LR')
        except:
            self.notice("{}\nMaybe you need to sudo apt-get install graphviz".format(sys.exc_info()[0]))
            raise

        return outfile
