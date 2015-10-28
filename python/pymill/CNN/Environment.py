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
import uuid
from Results import Results
from collections import OrderedDict


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

    def _callCopiedBin(self, cmd):
        bin = './' + os.path.basename(caffeBin())
        tb.notice('making a local copy of %s' % caffeBin())
        os.system('cp %s .' % caffeBin())

        ldd = tb.run('ldd %s' % caffeBin())
        caffeLib = None
        for line in ldd.split('\n'):
            match = re.match('\\s*libcaffe.so => (.*\.so)', line)
            if match:
                caffeLib = match.group(1)
                break
        if caffeLib is None:
            raise Exception('cannot find libcaffe.so dependency')

        tb.notice('making a local copy of %s' % caffeLib)
        os.system('cp %s .' % caffeLib)

        cmd = 'GLOG_logtostderr=%d LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH %s %s' % (not self._quiet, bin, cmd)
        if not self._silent:
            tb.notice('running "%s"' % cmd, 'run')
        tb.system(cmd)

    def train(self, solverFilename, logFile, weights=None):
        if weights is not None:
            weightOption = '-weights %s' % weights
        else:
            weightOption = ''

        self._callCopiedBin(Template('train -sighup_effect snapshot -solver $solverFilename $weightOption -gpu $gpu 2>&1 | tee -a $logFile').substitute({
            'solverFilename': solverFilename,
            'weightOption': weightOption,
            'gpu': self._gpus,
            'logFile': logFile
        }))

    def resume(self, solverFilename, solverstateFilename, logFile):
        self._callCopiedBin(Template(
            'train -sighup_effect snapshot -solver $solverFilename -snapshot $solverstateFilename -gpu $gpu 2>&1 | tee -a $logFile').substitute(
            {
                'solverFilename': solverFilename,
                'solverstateFilename': solverstateFilename,
                'gpu': self._gpus,
                'logFile': logFile
            }))

    def run(self, caffemodelFilename,iterations):
        self._callBin(Template('test -model $caffemodelFilename --iterations $iterations -gpu $gpu 2>&1').substitute({
            'caffemodelFilename': caffemodelFilename,
            'iterations': iterations,
            'gpu': self._gpus
        }))

    def test(self, caffemodelFilename, protoFilename, iterations, logFile):
        self._callBin(Template(
            'test -weights $caffemodelFilename -model $protoFilename -iterations $iterations -gpu $gpu 2>&1 | tee $logFile').substitute(
            {
                'caffemodelFilename': caffemodelFilename,
                'protoFilename': protoFilename,
                'gpu': self._gpus,
                'iterations': iterations,
                'logFile': logFile
            }))


class PythonBackend(BinaryBackend):
    def __init__(self, gpus='0', quiet=False, silent=False):
        self._gpus = gpus
        self._silent = silent
        self._quiet = quiet
        self.solver = None

    def train(self, solverFilename, logFile, weights=None):
        # hackaround: this backend doesn't log to one file but to a db in a subdirectory

        import caffe
        caffe.setup_teeing(logFile)

        from pymill.CNN import MillSolver as ms
        self.solver = ms.MillSolver(solver_def=solverFilename, weights=weights, gpus=[int(x) for x in self._gpus.split(',')])
        self.solver.run_solver()

    def resume(self, solverFilename, solverstateFilename, logFile):
        # hackaround: this backend doesn't log to one file but to a db in a subdirectory

        import caffe
        caffe.setup_teeing(logFile)

        log_dir = os.path.abspath(logFile)
        from pymill.CNN import MillSolver as ms
        self.solver = ms.MillSolver(solver_def=solverFilename, solver_state=solverstateFilename, gpus=[int(x) for x in self._gpus.split(',')])
        self.solver.run_solver()

    def get_log_dir_from_filename(self, logFile):
        log_dir = os.path.abspath(logFile)
        log_dir = os.path.join(log_dir[1:log_dir.rfind('/')], 'log')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)


class Environment:
    class Measure:
        def __init__(self, type = None, position=None):
            self._type = type
            self._position = position

        def parseFrom(self, line):
            parts = line.split(':')
            if len(parts) == 1:
                self._type = parts[0].strip()
            elif len(parts) == 2:
                self._type = parts[0].strip()
                self._position = parts[1].strip()
            else:
                raise Exception('invalid measure: %s' % measure)

        def type(self): return self._type

        def position(self): return self._position

        def __str__(self):
            if self._position: return self._type + ':' + self._position
            return self._type

    class Parameters:
        def __init__(self, env):
            self._lines = ''
            self._gpuArch = 'any'
            self._env = env
            self._task = None
            self._measures = []
            self._nameDepth = 1
            self._requiredMemory = 5*1024

        def read(self, filename):
            self._lines = open(filename).readlines()

            for line in self._lines:
                line = line.strip()
                if line.strip() == '': continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'gpu-arch':     self._gpuArch = value
                elif key == 'name':       self._env._name = value
                elif key == 'task':       self._task = value
                elif key == 'measure':
                    measure = Environment.Measure()
                    measure.parseFrom(value)
                    self._measures.append(measure)
                elif key == 'name-depth': self._nameDepth = int(value)
                elif key == 'required-memory': self._requiredMemory = int(value)
                else:
                    raise Exception('invalid entry in params.txt: %s' % key)

        def gpuArch(self): return self._gpuArch
        def task(self):
            if self._task is None: raise Exception('task not set in params.txt')
            return self._task
        def measures(self): return self._measures
        def nameDepth(self): return self._nameDepth
        def requiredMemory(self): return self._requiredMemory

    def __init__(self, path='.', backend=BinaryBackend(), unattended=False, silent=False):
        self._path = os.path.abspath(path)
        self._backend = backend
        self._unattended = unattended
        self._silent = silent

    def trainDir(self): return self._trainDir

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

        tb.system('rm -f %s/scratch/current' % self._path)
        tb.system('ln -s %s %s/scratch/current' % (self._scratchDir, self._path))

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
            if os.system('python -B %s %s > %s' % (inFile, args, prototxt)) != 0:
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

        self._logDir = self._path + '/training/log'
        self._trainDir = self._path + '/training'
        self._scratchDir = self._path + '/scratch/%s' % uuid.uuid4()
        self._jobDir = self._path + '/jobs'

        self._modelFiles = iterFiles('.caffemodel', self._trainDir)
        self._stateFiles = iterFiles('.solverstate', self._trainDir)
        self._logFiles = logFiles(self._logDir)

        self._trainProto = self.findProto('train')
        self._solverProto = self.findProto('solver')

        self._logFile = self._path + '/training/log.txt'

        self._scratchLogFile = self._scratchDir + '/log.txt'

        self._params = Environment.Parameters(self)
        if os.path.isfile(self._path + '/params.txt'):
            self._params.read(self._path + '/params.txt')

        parts = list(reversed(os.path.normpath(self._path).split('/')))
        parts = list(reversed(parts[0:self.params().nameDepth()]))
        self._name = '/'.join(parts)

    def determineTestDatasets(self):
        if self.params().task() is None: raise Exception('you need to specify a task in params.txt')

        from pymill.CNN.Definition.Dataset import getDatasetNames

        return getDatasetNames(self.params().task())

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
        log = Log(self._name, self._logFile)
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
        os.system('rm -f %s/*.pyc' % (self._path))

        if self.haveTrainDir():
            self.notice('removing training', 'del')
            os.system('rm -rf %s' % self._trainDir)

        if self.haveScratchDir():
            self.notice('removing scratch', 'del')
            os.system('rm -rf %s/scracth' % self._path)

        if self.haveJobDir():
            self.notice('removing jobs', 'del')
            os.system('rm -rf %s' % self._jobDir)

        self.sweep()

    def sweep(self):
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

        self.makeTrainingPrototxt(self._trainProto)
        return self.makeTrainingPrototxt(self._solverProto)

    def displayBlobSummary(self, logfile):
        log = Log(self._name, logfile)
        log.displayBlobSummary()

    def blobSummary(self):
        log = Log(self._name, self._logFile)
        log.displayBlobSummary()

    def train(self, weights=None, blobSummary=False):
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

        if blobSummary:
            self.displayBlobSummary(self._logFile)

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

    def _saveTestResults(self, iter, dataset):
        log = Log(self._name, self._scratchLogFile)

        results = OrderedDict()
        for measure in self.params().measures():
            value = log.getAssignment(str(measure))
            Results(self._path).update(iter, dataset, self.params().task(), measure.type(), measure.position(), value)
            results[str(measure)] = value

        return results

    def test(self, iter, output=False, definition=None, vars={}, num_iter=-1):
        modelFile, iter = self.getModelFile(iter)

        if output:
            vars['output'] = True
            vars['prefix'] = iter

        self.makeScratchDir()

        if definition is None: definition = 'test'
        proto = self.findProto(definition)

        if output and 'dataset' in vars:
            outPath = '%s/output_%d_%s' % (self._path, iter, vars['dataset'])
            if os.path.isdir(outPath):
                if self._unattended or tb.queryYesNo('Output folder %s exists, do you want to delete it first?' % os.path.basename(outPath)):
                    os.system('rm -rf %s' % outPath)

        finalProto = self.makeScratchPrototxt(proto, vars)
        solverProto = self.makeScratchPrototxt(self._solverProto, vars)

        self.notice('testing snapshot iteration %d for %d iterations...' % (iter, num_iter), 'notice')
        os.chdir(self._path)
        self._backend.test(caffemodelFilename=modelFile, protoFilename=finalProto, iterations=num_iter, logFile=self._scratchLogFile)

        if output and 'dataset' in vars:
            outPath = '%s/output_%d_%s' % (self._path, iter, vars['dataset'])
            if os.path.isdir(outPath):
                logFile = '%s/log.txt' % outPath
                print 'saving log to %s', logFile
                os.system('cp %s %s' % (self._scratchLogFile, logFile))

        if 'dataset' in vars:
            self._saveTestResults(iter,vars['dataset'])

        print 'Iteration was %d' %iter

    def runTests(self, iter, datasets, output=False, definition=None, vars={}):
        if isinstance(datasets,str): datasets = datasets.split(',')
        if datasets is None: datasets = self.determineTestDatasets()

        # Fix iteration here (use same for all tests)
        _, iter = self.getModelFile(iter)

        try:
            results = []
            for dataset in datasets:
                self._scratchLogFile = '%s/log-%s.txt' % (self._scratchDir, dataset)
                vars['dataset'] = dataset
                self.test(iter, output, definition, vars)

                values = self._saveTestResults(iter, dataset)
                results.append((dataset, values))
        finally:
            print
            print 'Results of net %s (%s) for iteration %d:' % (self.name(), self.params().task(), iter)
            print '--------------------------------------------------------------------------------'
            for result in results:
                for measure, value in result[1].iteritems():
                    print '%30s: %15s = %5.3f' % (result[0], measure, value)
            print '--------------------------------------------------------------------------------'
            print

    def plot(self, select):
        if not self.haveLogFile():
            raise Exception('logfile doesn\'t exist')

        log = Log(self._name, self._logFile)
        log.plot(select)

    def plotLR(self):
        if not self.haveLogFile():
            raise Exception('logfile doesn\'t exist')

        log = Log(self._name, self._logFile)
        log.plotlr()

    def compare(self, networks, losses):
        folders = [dir for dir in os.listdir('.') if os.path.isdir(dir) and not dir.startswith('.')]
        networks = tb.wildcardMatch(folders, networks)

        logs = []
        measureNames = []
        for net in networks:
            logfile = '%s/training/log.txt' % net
            print 'reading %s' % logfile
            logs.append(Log(net, logfile))
            for name in logs[-1].measureNames():
                if name not in measureNames: measureNames.append(name)

        if losses is not None:
            selectedNames = tb.unique(tb.wildcardMatch(measureNames, losses))
        else:
            selectedNames = tb.unique(measureNames)

        print 'comparing networks:'
        for net in networks: print "   ", net
        print 'comparing losses: '
        for name in selectedNames: print "   ", name

        Log.plotComparison(selectedNames, logs)

    def viewFilters(self, iter):
        self.prepareTraining()
        prototxt = self._trainDir + '/train.prototxt'
        modelFile, iter = self.getModelFile(iter)

        os.environ['LD_LIBRARY_PATH']="/misc/lmbraid17/sceneflownet/common/programs/torch/install/lib:/usr/lib/x86_64-linux-gnu:/misc/lmbraid17/sceneflownet/common/software-root/lib:/home/ilge/dev/hackathon-caffe2/build/lib:/misc/software-lin/Qt-5.3.2/5.3/gcc_64/lib:/misc/lmbraid17/sceneflownet/common/programs/torch/install/lib:/usr/lib/x86_64-linux-gnu:/misc/lmbraid17/sceneflownet/common/software-root/lib:/home/ilge/dev/hackathon-caffe2/build/lib:/misc/software-lin/Qt-5.3.2/5.3/gcc_64/lib::/home/ilge/lib:/misc/software-lin/lmbsoft/openni-1.5.2.23-x86_64/usr/lib:/misc/software-lin/lmbsoft/glog/lib:/misc/software-lin/lmbsoft/mkl/lib:/misc/software-lin/lmbsoft/mkl/lib/intel64:/misc/software-lin/lmbsoft/cuda-6.5.14-x86_64/lib64:/misc/software-lin/lmbsoft/cuda-6.0.37-x86_64/lib64:/misc/student/mayern/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Lib:/home/ilge/lib:/misc/software-lin/lmbsoft/openni-1.5.2.23-x86_64/usr/lib:/misc/software-lin/lmbsoft/glog/lib:/misc/software-lin/lmbsoft/mkl/lib:/misc/software-lin/lmbsoft/mkl/lib/intel64:/misc/software-lin/lmbsoft/cuda-6.5.14-x86_64/lib64:/misc/software-lin/lmbsoft/cuda-6.0.37-x86_64/lib64:/misc/student/mayern/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Lib"
        os.environ['PATH']="/home/ilge/bin:/home/ilge/dev/pymill/bin:/misc/lmbraid17/sceneflownet/common/programs/torch/install/bin:/misc/lmbraid17/sceneflownet/common/software-root/bin:/misc/lmbraid17/sceneflownet/ilge/hackathon-caffe2/python/pymill/bin:/misc/software-lin/Qt-5.3.2/5.3/gcc_64/bin:/home/ilge/bin:/home/ilge/dev/pymill/bin:/misc/lmbraid17/sceneflownet/common/programs/torch/install/bin:/misc/lmbraid17/sceneflownet/common/software-root/bin:/misc/lmbraid17/sceneflownet/ilge/hackathon-caffe2/python/pymill/bin:/misc/software-lin/Qt-5.3.2/5.3/gcc_64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/misc/software-lin/lmbsoft/cuda-6.5.14-x86_64/bin:/misc/software-lin/matlabR2013a/bin:/home/ilge/data/caffe/matching/bin:/misc/lmbraid15/hackathon/common/flo-results/bin:/misc/lmbraid17/sceneflownet/common/data_tools:/misc/software-lin/lmbsoft/cuda-6.5.14-x86_64/bin:/misc/software-lin/matlabR2013a/bin:/home/ilge/data/caffe/matching/bin:/misc/lmbraid15/hackathon/common/flo-results/bin:/misc/lmbraid17/sceneflownet/common/data_tools"

        tb.system('/home/ilge/bin/weight-viewer %s %s' % (prototxt, modelFile))

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
            if os.path.isdir('%s/%s' % (source,f)) and f.startswith('output'): continue

            tb.system('cp -r %s %s/%s %s' % ('' if self._silent else '-v', source, f, target))

    def execute(self, file, iter):
        self.makeScratchDir()
        finalProto = self.makeScratchPrototxt(file)

        self.notice('running %s for %d iterations ...' % (file, iter), 'notice')
        os.chdir(self._path)
        self._backend.run(finalProto, iter)

    def draw(self):
        from google.protobuf import text_format
        import caffe, caffe.draw
        from caffe.proto import caffe_pb2

        self.prepareTraining()
        prototxt = self._trainDir + '/train.prototxt'

        outfile = prototxt.replace('prototxt', 'png') #self._scratchDir + '/../%s.png' % os.path.basename(prototxt).replace('.prototxt', '')
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt).read(), net)
        self.notice('drawing net to %s' % outfile)
        try:
            caffe.draw.draw_net_to_file(net, outfile, 'LR')
        except:
            self.notice("{}\nMaybe you need to sudo apt-get install graphviz".format(sys.exc_info()[0]))
            raise

        return outfile
