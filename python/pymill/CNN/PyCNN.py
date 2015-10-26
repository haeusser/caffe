#!/usr/bin/python
# PYTHON_ARGCOMPLETE_OK

# How to activate autocompletion:
# First do
#   $ pip install --user argcomplete (or without user if you are root)
# Then install the automcompletion script.
#   $ activate-global-python-argcomplete
#
# If you are not root, then install locally:
#   $ activate-global-python-argcomplete --user
# and make sure you source the resulting script in your .bashrc:
#   . ~/.bash_completion.d/python-argcomplete.sh

import os
import sys
## (mayern) remove my .local folder from the PYTHONPATH -->
EVILPATH = '/home/mayern/.local/lib/python2.7/site-packages'
if EVILPATH in sys.path:
  sys.path.remove(EVILPATH)
  import matplotlib.pyplot as plt
  sys.path.append(EVILPATH)
else:
  import matplotlib.pyplot as plt
## <-- (mayern) remove my .local folder from the PYTHONPATH
from string import Template
from termcolor import colored
import argparse
import argcomplete
import numpy as np
from pymill import Toolbox as tb
from pymill import CNN as CNN
import re
from Environment import Environment
from Environment import PythonBackend
from Environment import BinaryBackend
from Environment import caffeBin
import time
import signal
import caffe

def sigusr1(signum, stack):
    print 'pycnn: got signal SIGUSR1'

signal.signal(signal.SIGUSR1, sigusr1)

# make files writeable for group lmb_hackathon !
os.umask(002)

def parseParameters(params):
    d = {}

    for p in params:
        if not '=' in p:
            raise Exception('Parameter %s is not of the form key=value' % p)
        k, v = p.split('=')
        d[k] = v

    return d

def runOnCluster(env, node, gpus, background,insertLocal=True, trackJob=True):
    gpuArch = env.params().gpuArch()
    if node is not None: tb.notice('Forwarding job to cluster node %s with %d gpu(s) which are of type %s' % (node, gpus, gpuArch),'info')
    else:                tb.notice('Forwarding job to cluster with %d gpu(s) which are of type %s' % (gpus, gpuArch),'info')

    env.makeJobDir()

    currentId = '%s/current_id' %env.jobDir()
    if trackJob and os.path.exists(currentId):
        raise Exception('%s exists, there seems to be a job already running' % currentId)

    sysargs = sys.argv
    if insertLocal:
        sysargs.insert(1,'--execute')
    cmd = ' '.join(sysargs)
    home = os.environ['HOME']

    if args.backend == 'python':
        training = os.path.abspath('training')
        cmd = 'LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH PYTHONPATH=%s:$PYTHONPATH %s' % (training, training, cmd)

    qsubCommandFile = '%s/%s-%s.sh' % (env.jobDir(), env.name().replace('/','_'), time.strftime('%d.%m.%Y-%H:%M:%S'))

    epilogueScript = '%s/epilogue.sh' %env.jobDir()
    open(epilogueScript, 'w').write("#!/bin/bash\ncd $path\nrm -f jobs/current_id\n")

    if trackJob: saveIdCommand = 'echo $$PBS_JOBID > jobs/current_id'
    else:        saveIdCommand = ''

    script = Template(
    '#!/bin/bash\n'
    '\n'
    'umask 0002\n'
    'echo -e "\e[30;42m --- running on" `hostname` "--- \e[0m"\n'
    'cd "$path"\n'
    '$saveIdCommand\n'
    'trap "echo got SIGHUP" SIGHUP\n'
    'trap "echo got SIGUSR1" USR1\n'
    '$command\n'
    'echo done\n'
    'rm -f jobs/current_id\n'
    ).substitute(path=env.path(), command=cmd, saveIdCommand=saveIdCommand)

    open(qsubCommandFile,'w').write(script)
    tb.system('chmod a+x "%s"' % qsubCommandFile)

    qsub = 'qsub -l nodes=%s:gpus=%d%s,mem=%dmb,walltime=240:00:00 %s -q gpujob -d %s %s -N %s -T %s' % (
        node if node is not None else '1',
        gpus,
        (':' + gpuArch) if gpuArch!='any' else '',
        env.params().requiredMemory(),
        '-I -x' if not background else '',
        env.path(),
        qsubCommandFile,
        env.name(),
        epilogueScript
    )

    if background:
        print 'job name: %s' % os.path.basename(qsubCommandFile)
        qsub += ' -j oe -o %s' % (env.jobDir())

    tb.notice("lmbtorque: running %s" % qsub, 'run')

    if not background: tb.system('ssh lmbtorque "umask 0002; cd %s; %s;  rm -f jobs/current_id"' % (env.path(), qsub))
    else:              tb.system('ssh lmbtorque "umask 0002; %s"' % (qsub))
    sys.exit(0)


parser = tb.Parser(
    prog = 'pycnn',
    usage = (
        "pycnn [options] <command> [<args>]\n"
        "\n"
        "PyCNN is used to manage caffe models. Run it from the model folder \n"
        "or specify the path using '--path PATH'. The model folder should contain:\n"
        "solver[.py|.prototmp|.prototxt] and train[.py|.prototmp|.prototxt]"
        "\n"
        "note:\n"
        "The operations train, test and continue will be executed on the\n"
        "cluster if --local is not specified. run is executed locally unless\n"
        "--cluster is specified. Cluster jobs are interactive unless\n"
        "you pass --background.\n"
        "\n"
        "note:\n"
        "The general options listed below need to be specified before <command>.\n"
    )
)

parser.add_argument('--verbose',       help='verbose', action='store_true')
parser.add_argument('--path',          help='model path (default=.)', default='.')
parser.add_argument('--unattended',    help='always assume Y as answer (dangerous)', action='store_true')
parser.add_argument('--yes',           help='same as unattended', action='store_true')
parser.add_argument('--backend',       help='backend to use (default=python)', choices=('binary','python'), default='binary')
parser.add_argument('--local',         help='run on local machine', action='store_true')
parser.add_argument('--background',    help='run on cluster in background', action='store_true')
parser.add_argument('--node',          help='run on a specific node', default=None)
parser.add_argument('--gpus',          help='gpus to use: N (default=1)', default=1, type=int)
parser.add_argument('--gpu-id',        help='outside cluster: gpu ID to use (default=0)', default=None, type=int)
parser.add_argument('--cluster',       help='run on cluster interatively', action='store_true')
parser.add_argument('--quiet',         help='suppress caffe output', action='store_true')
parser.add_argument('--silent',        help='suppress all output', action='store_true')
parser.add_argument('--execute',       help='(used internally only)', action='store_true')

subparsers = parser.add_subparsers(dest='command', prog='pycnn')

# train
subparser = subparsers.add_parser('train', help='train a network from scratch')
subparser.add_argument('--weights',    help='caffe model file to initialize from', default='')
subparser.add_argument('--blob-sum',   help='display blob summary at the end', action='store_true')

# test
subparser = subparsers.add_parser('test', help='test a network')
subparser.add_argument('--iter',       help='iteration of .caffemodel to use', default=-1, type=int)
subparser.add_argument('--num-iter',   help='number of iterations to run (default=auto)', default=-1, type=int)
subparser.add_argument('--def',        help='custom test definition (default test.proto*/.py)', default=None)
subparser.add_argument('--output',     help='output images to folder output_...', action='store_true')
subparser.add_argument('param',        help='parameter to network', nargs='*')

# run-tests
subparser = subparsers.add_parser('run-tests', help='run multiple tests')
subparser.add_argument('--iter',       help='iteration of .caffemodel to use', default=-1, type=int)
subparser.add_argument('--datasets',   help='list of datasets (default=all)', default=None)
subparser.add_argument('--def',        help='custom test definition (default test.proto*/.py)', default=None)
subparser.add_argument('--output',     help='output images to folder output_...', action='store_true')
subparser.add_argument('param',        help='parameter to network', nargs='*')

# continue
subparser = subparsers.add_parser('continue', help='continue training from last saved (or specified) iteration')
subparser.add_argument('--iter',       help='iteration from which to continue (default=last)', default=-1, type=int)

# run
subparser = subparsers.add_parser('run', help='exectue a .prototxt or .prototmp file')
subparser.add_argument('file',         help='filename of .prototxt or .prototmp file')
subparser.add_argument('--iter',       help='number of iterations to run', default=-1, type=int)

# clean
subparser = subparsers.add_parser('clean', help='delete .caffemodel, .solverstate and log files')
subparser.add_argument('--iter', help='delete only everything after ITER', default=-1, type=int)

# sweep
subparser = subparsers.add_parser('sweep', help='delete output folders')

# sanitize
subparser = subparsers.add_parser('sanitize', help='delete everything that was created by pycnn')

# plot
subparser = subparsers.add_parser('plot', help='plot losses and accuracies')
subparser.add_argument('--select', help='selection of measures, e.g. test_*,train_*', default='')

# plot-lr
subparser = subparsers.add_parser('plot-lr', help='plot learning rate')

# plot-test
subparser = subparsers.add_parser('plot-test', help='plot test losses')

# compare
subparser = subparsers.add_parser('compare', help='compare the losses of some networks')
subparser.add_argument('networks', help='comma separated list of networks')
subparser.add_argument('losses',   help='comma separated list of loss names', default=None)

# view
subparser = subparsers.add_parser('view', help='view weights')
subparser.add_argument('--iter', help='iteration of .caffemodel (default=last)', default=-1, type=int)

# draw
subparser = subparsers.add_parser('draw', help='draw model diagram')

# copy
sub_parser = subparsers.add_parser('copy', help='copy a model')
sub_parser.add_argument('source', help='source directory')
sub_parser.add_argument('target', help='target directory')
sub_parser.add_argument('--with-snapshot', help='last snapshot', action='store_true')
sub_parser.add_argument('--iter', help='iteration of snapshot (default=last)', default=-1, type=int)

# snapshot
sub_parser = subparsers.add_parser('snapshot', help='connect to current process and request snapshot')

# snapshot
sub_parser = subparsers.add_parser('blob-sum', help='blob summary of current trainig log')

# autocomplete very slow for some reason
#argcomplete.autocomplete(parser)

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(2)

args = parser.parse_args()
tb.verbose = args.verbose

args.unattended = args.yes

if args.background: args.unattended = True
if args.execute: args.local = True

gpuIds = ''
if args.gpu_id is not None:
    gpuIds = '%d' % args.gpu_id
else:
    for i in range(0, args.gpus):
        gpuIds += ',%d' % i
    gpuIds = gpuIds[1:]

if args.backend == 'binary':
    backend = BinaryBackend(gpuIds, args.quiet, args.silent)
else:
    backend = PythonBackend(gpuIds, args.quiet, args.silent)
    if args.execute:
        print 'using caffe module from: %s' % caffe.__file__
        print 'using caffe._caffe module from: %s' % caffe._caffe.__file__

        ldd = tb.run('ldd %s' % caffe._caffe.__file__)
        caffeLib = None
        for line in ldd.split('\n'):
            match = re.match('\\s*libcaffe.so => (.*\.so)', line)
            if match:
                caffeLib = match.group(1)
                break
        if caffeLib is None:
            raise Exception('cannot find libcaffe.so dependency')
        print 'using caffe from %s' % caffeLib


env = Environment(args.path, backend, args.unattended, args.silent)
if args.command != 'copy' and args.command != 'compare': env.init()

def checkJob():
    currentId = '%s/current_id' % env.jobDir()
    if not os.path.exists(currentId):
        raise Exception('cannot find %s, no job seems to be running.' % currentId)
    return open(currentId).read().strip()

def checkNoJob():
    currentId = '%s/current_id' % env.jobDir()
    if os.path.exists(currentId):
        raise Exception('%s exists, there seems to be a job running' % currentId)

def preparePythonBackend():
    os.system('mkdir -p training')
    folder = os.path.dirname(caffe.__file__)
    print 'copying %s to training' % folder
    os.system('cp %s training -r' % folder)

    ldd = tb.run('ldd %s' % caffe._caffe.__file__)
    caffeLib = None
    for line in ldd.split('\n'):
        match = re.match('\\s*libcaffe.so => (.*\.so)', line)
        if match:
            caffeLib = match.group(1)
            break
    if caffeLib is None:
        raise Exception('cannot find libcaffe.so dependency')
    print 'copying %s to training' % caffeLib

    os.system('cp %s %s' % (caffeLib, env.trainDir()))

# local operations
if   args.command == 'clean':
    checkNoJob()
    env.clean(args.iter)
    sys.exit(0)
if   args.command == 'sanitize':
    checkNoJob()
    env.sanitize()
    sys.exit(0)
if   args.command == 'sweep':
    env.sweep()
    sys.exit(0)
elif args.command == 'plot':
    env.plot(args.select)
    sys.exit(0)
elif args.command == 'plot-lr':
    env.plotLR()
    sys.exit(0)
elif args.command == 'plot-test':
    env.plot(select='test*')
    sys.exit(0)
elif args.command == 'compare':
    env.compare(args.networks, args.losses)
    sys.exit(0)
elif args.command == 'view':
    env.view(args.iter)
    sys.exit(0)
elif args.command == 'copy':
    env.copy(args.source, args.target, args.with_snapshot, args.iter)
    sys.exit(0)
elif args.command == 'draw':
    os.system('gwenview %s &' % env.draw())
    sys.exit(0)
elif args.command == 'snapshot':
    id = checkJob()
    os.system('ssh lmbtorque "qsig -s SIGHUP %s"' % id)
    sys.exit(0)
elif args.command == 'blob-sum':
    env.blobSummary()
    sys.exit(0)

# gpu operations
if   args.command == 'train':
    if not args.execute: checkNoJob()
    if args.backend == 'python' and not args.execute:
        preparePythonBackend()
    if args.local:
        env.train(args.weights, args.blob_sum)
        sys.exit(0)
    else:
        runOnCluster(env, args.node, args.gpus, args.background)
elif args.command == 'test':
    if args.backend == 'python' and not args.execute: preparePythonBackend()
    if args.local:
        env.test(args.iter, args.output, getattr(args,'def'), parseParameters(args.param), args.num_iter)
        sys.exit(0)
    else:
        runOnCluster(env, args.node, args.gpus, args.background, trackJob=False)
elif args.command == 'run-tests':
    if args.local:
        env.runTests(args.iter, args.datasets, args.output, getattr(args,'def'), parseParameters(args.param))
        sys.exit(0)
    else:
        runOnCluster(env, args.node, args.gpus, args.background, trackJob=False)
elif args.command == 'continue':
    if not args.execute: checkNoJob()
    if args.local:
        env.resume(args.iter)
        sys.exit(0)
    else:
        runOnCluster(env, args.node, args.gpus, args.background)
elif args.command == 'run':
    if args.cluster:
        runOnCluster(env, args.node, args.gpus, args.background,False, trackJob=False)
    else:
        env.execute(args.file, args.iter)
        sys.exit(0)









