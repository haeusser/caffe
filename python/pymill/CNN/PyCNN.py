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
from string import Template
from termcolor import colored
import argparse
import argcomplete
import numpy as np
import matplotlib.pyplot as plt
from pymill import Toolbox as tb
from pymill import CNN as CNN
import re
from Environment import Environment
from Environment import PythonBackend
from Environment import BinaryBackend
import time
import signal

def sigusr1(signum, stack):
    print 'pycnn: got signal SIGUSR1'

signal.signal(signal.SIGUSR1, sigusr1)

# make files writeable for group lmb_hackathon !
os.umask(002)
 
def runOnCluster(env, node, gpus, background,insertLocal=True):
    gpuArch = env.params().gpuArch()
    if node is not None: tb.notice('Forwarding job to cluster node %s with %d gpu(s) which are of type %s' % (node, gpus, gpuArch),'info')
    else:                tb.notice('Forwarding job to cluster with %d gpu(s) which are of type %s' % (gpus, gpuArch),'info')

    env.makeJobDir()

    currentId = '%s/current_id' %env.jobDir()
    if os.path.exists(currentId):
        raise Exception('%s exists, there seems to be a job already running' % currentId)

    sysargs = sys.argv
    if insertLocal:
        sysargs.insert(1,'--execute')
    cmd = ' '.join(sysargs)
    home = os.environ['HOME']

    if args.background: qsubCommandFile = '%s/%s-%s.sh' % (env.jobDir(), env.name().replace('/','_'), time.strftime('%d.%m.%Y-%H:%M:%S'))
    else:               qsubCommandFile = '%s/interactive.sh' % env.jobDir()

    epilogueScript = '%s/epilogue.sh' %env.jobDir()
    open(epilogueScript, 'w').write("#!/bin/bash\ncd $path\nrm -f jobs/current_id\n")

    script = Template(
    '#!/bin/bash\n'
    '\n'
    'umask 0002\n'
    'echo -e "\e[30;42m --- running on" `hostname` "--- \e[0m"\n'
    'cd "$path"\n'
    'echo $$PBS_JOBID > jobs/current_id\n'
    'trap "echo got SIGHUP" SIGHUP\n'
    'trap "echo got SIGUSR1" USR1\n'
    '$command\n'
    'echo done\n'
    'rm -f jobs/current_id\n'
    ).substitute(path=env.path(), command=cmd)

    open(qsubCommandFile,'w').write(script)
    tb.system('chmod a+x "%s"' % qsubCommandFile)

    qsub = 'qsub -l nodes=%s:gpus=%d%s,walltime=240:00:00 %s -q gpujob -d %s %s -N %s -T %s' % (
        node if node is not None else '1',
        gpus,
        (':' + gpuArch) if gpuArch!='any' else '',
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
parser.add_argument('--backend',       help='backend to use (default=binary)', choices=('binary','python'))
parser.add_argument('--local',         help='run on local machine', action='store_true')
parser.add_argument('--background',    help='run on cluster in background', action='store_true')
parser.add_argument('--node',          help='run on a specific node', default=None)
parser.add_argument('--gpus',          help='gpus to use: N (default=1)', default=1, type=int)
parser.add_argument('--cluster',       help='run on cluster interatively', action='store_true')
parser.add_argument('--quiet',         help='suppress caffe output', action='store_true')
parser.add_argument('--silent',        help='suppress all output', action='store_true')
parser.add_argument('--execute',       help='(used internally only)', action='store_true')

subparsers = parser.add_subparsers(dest='command', prog='pycnn')

# train
subparser = subparsers.add_parser('train', help='train a network from scratch')
subparser.add_argument('--weights',    help='caffe model file to initialize from', default='')

# test
subparser = subparsers.add_parser('test', help='test a network')
subparser.add_argument('--iter',       help='iteration of .caffemodel to use', default=-1, type=int)
subparser.add_argument('--variant',    help='test variant', default=None)
subparser.add_argument('--output',     help='output images to folder output_...', action='store_true')

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

# sanitize
subparser = subparsers.add_parser('sanitize', help='delete everything that was created by pycnn')

# plot
subparser = subparsers.add_parser('plot', help='plot losses and accuracies')
subparser.add_argument('--select', help='selection of measures, e.g. test_*,train_*', default='')

# plotlr
subparser = subparsers.add_parser('plotlr', help='plot learning rate')

# view
subparser = subparsers.add_parser('view', help='view weights')
subparser.add_argument('--iter', help='iteration of .caffemodel (default=last)', default=-1, type=int)

# draw
subparser = subparsers.add_parser('draw', help='draw model diagram')
subparser.add_argument('file', help='filename of .prototxt or .prototmp file')

# copy
sub_parser = subparsers.add_parser('copy', help='copy a model')
sub_parser.add_argument('source', help='source directory')
sub_parser.add_argument('target', help='target directory')
sub_parser.add_argument('--copy-snapshot', help='last snapshot', action='store_true')
sub_parser.add_argument('--iter', help='iteration of snapshot (default=last)', default=-1, type=int)

# snapshot
sub_parser = subparsers.add_parser('snapshot', help='connect to current process and request snapshot')

# autocomplete very slow for some reason
argcomplete.autocomplete(parser)

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(2)

args = parser.parse_args()
tb.verbose = args.verbose

args.unattended = args.yes

if args.background: args.unattended = True
if args.execute: args.local=True

gpuIds = ''
for i in range(0, args.gpus):
    gpuIds += ',%d' % i
gpuIds = gpuIds[1:]
if args.backend == 'python': backend = PythonBackend(gpuIds, args.quiet, args.silent)
else:                        backend = BinaryBackend(gpuIds, args.quiet, args.silent)

env = Environment(args.path, backend, args.unattended, args.silent)
if args.command != 'copy': env.init()

def checkJob():
    currentId = '%s/current_id' % env.jobDir()
    if not os.path.exists(currentId):
        raise Exception('cannot find %s, no job seems to be running.' % currentId)
    return open(currentId).read().strip()

def checkNoJob():
    currentId = '%s/current_id' % env.jobDir()
    if os.path.exists(currentId):
        raise Exception('%s exists, there seems to be a job running' % currentId)

# local operations
if   args.command == 'clean':
    checkNoJob()
    env.clean(args.iter)
    sys.exit(0)
if   args.command == 'sanitize':
    checkNoJob()
    env.sanitize()
    sys.exit(0)
elif args.command == 'plot':
    env.plot(args.select)
    sys.exit(0)
elif args.command == 'plotlr':
    env.plotLR()
    sys.exit(0)
elif args.command == 'view':
    env.view(args.iter)
    sys.exit(0)
elif args.command == 'copy':
    env.copy(args.source, args.target, args.copy_snapshot, args.iter)
    sys.exit(0)
elif args.command == 'draw':
    sys.exit(0)
    os.system('gwenview %s &' % env.draw(args.file))
elif args.command == 'snapshot':
    id = checkJob()
    os.system('ssh lmbtorque "qsig -s SIGHUP %s"' % id)
    sys.exit(0)

# gpu operations
if   args.command == 'train':
    if not args.execute: checkNoJob()
    if args.local:
        env.train(args.weights)
        sys.exit(0)
    else:
        runOnCluster(env, args.node, args.gpus, args.background)
elif args.command == 'test':
    if args.local:
        env.test(args.iter, args.output, args.variant)
        sys.exit(0)
    else:
        runOnCluster(env, args.node, args.gpus, args.background)
elif args.command == 'continue':
    if not args.execute: checkNoJob()
    if args.local:
        env.resume(args.iter)
        sys.exit(0)
    else:
        runOnCluster(env, args.node, args.gpus, args.background)
elif args.command == 'run':
    if args.cluster:
        runOnCluster(env, args.node, args.gpus, args.background,False)
    else:
        env.execute(args.file, args.iter)
        sys.exit(0)









