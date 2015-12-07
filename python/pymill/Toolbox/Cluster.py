#!/usr/bin/python

import os
import shutil
from Notice import notice

class Job:
    def __init__(self):
        self._id = -1
        self._commands = []
        self._mem = 1024
        self._cores = 1
        self._walltimeHours = 0
        self._walltimeMinutes = 10
        self._log = '/dev/null'
        self._gpu = False

    def id(self): return self._id
    def setId(self, id): self._id = id

    def gpu(self): return self._gpu
    def setGpu(self, gpu): self._gpu = gpu

    def log(self): return self._log
    def setLog(self,log): self._log = log

    def commands(self): return self._commands
    def addCommand(self, path, cmd):
        self._commands.append((path, cmd))

    def mem(self): return self._mem
    def setMem(self, mem): self._mem = mem

    def cores(self): return self._cores
    def setCores(self, cores): self._cores = cores

    def walltime(self, factor=1):
        mins = factor*(self._walltimeHours*60 + self._walltimeMinutes)

        m = mins % 60
        h = mins / 60

        return '%02d:%02d:00' % (h, m)

    def setWalltime(self, hrs,mns): self._walltimeHours=hrs; self._walltimeMinutes=mns

class Queue:
    _currentId = 0

    def __init__(self,name=None):
        self._packets = []
        self._jobs = []

        if name:
            self._name = name
            self._path = '/home/ilge/jobs/%s' % self._name
        else:
            i =0
            while True:
                i += 1
                self._name = '%03d' % i
                self._path = '/home/ilge/jobs/%s' % self._name
                if not os.path.isdir(self._path):
                    break

            notice('using queue name <%s>' % self._name)

        if os.path.isdir(self._path):
            notice('removing queue <%s>' % self._path, 'del')
            shutil.rmtree(self._path)

        os.makedirs(self._path)

    def postJob(self, job):
        if not len(job.commands()):
            return

        if job.id() == -1:
            job.setId(self._currentId)
            self._currentId += 1

        self._jobs.append(job)

    def finishPacket(self):
        self._packets.append(self._jobs)
        self._jobs = []

    def submit(self,local=False, cores=1):
        cwd = os.getcwd()

        if len(self._jobs):
            self.finishPacket()

        jobs = []
        for packet in self._packets:
            jobs += packet

        group = 1
        if len(jobs) > 1100:
            group = len(jobs) / 500
        walltime = jobs[0].walltime(group)

        notice('using group size %d and walltime %s' % (group, walltime))

        f = open('%s/job.sh' % self._path, 'w')
        f.write('#!/bin/bash\n')
        f.write('#PBS -N %s\n' %  (self._name))
        f.write('#PBS -S /bin/bash\n')
        f.write('#PBS -l nodes=1:ppn=%d,mem=%dmb,walltime=%s\n' % (jobs[0].cores(), jobs[0].mem(), walltime))
        f.write('#PBS -m a\n')
        f.write('#PBS -M %s@cs.uni-freiburg.de\n' % 'ilge')
        f.write('#PBS -j oe\n')
        f.write('set -e\n')
        f.write('set -o pipefail\n')
        n = 0

        while len(jobs):
            f.write('if [ "$PBS_ARRAYID" == %s ]; then\n' % n )

            for i in range(0, group):
                if not len(jobs): break
                job = jobs.pop()

                currentPath = '/'.join(job.log().split('/')[:-1])
                f.write('   mkdir -p %s 2>&1\n' % currentPath)
                f.write('   echo "---------- job started" | tee -a %s\n' % job.log())
                f.write('   echo `date`: cd %s 2>&1 | tee -a %s\n' % (currentPath, job.log()))
                f.write('   cd %s 2>&1 | tee -a %s\n' % (currentPath, job.log()))
                f.write('   cd %s 2>&1\n' % (currentPath))

                for path, cmd in job.commands():
                    if path != currentPath:
                        f.write('   mkdir -p %s 2>&1 | tee -a %s\n' % (path, job.log()))
                        f.write('   cd %s 2>&1 | tee -a %s\n' % (path, job.log()))
                        f.write('   cd %s\n' % (currentPath))
                        f.write('   echo `date`: cd %s 2>&1 | tee -a %s\n' % (path, job.log()))
                        currentPath = path
                    f.write('   echo `date`: \'%s\' 2>&1 | tee -a %s\n' % (cmd.replace('\'', '\'"\'"\''), job.log()))
                    f.write('   %s 2>&1 | tee -a %s\n' % (cmd, job.log()))
                f.write('   echo `date`: "SUCCEEDED" | tee -a %s 2>&1\n' % job.log())
                f.write('   #--------------------------------------\n')

            f.write('   exit 0\n')
            f.write('fi\n')
            n += 1

        f.close()

        notice('queue <%s> contains %d jobs' % (self._name, n))
        if local:
            notice('computing locally')

            for i in range(0,n):
                print os.system('PBS_ARRAYID=%d bash %s/job.sh' % (i,self._path)),

        else:
            notice('submitting queue <%s> to cluster' % self._name)
            os.system('ssh lmbtorque "cd /home/ilge/jobs; dir>/dev/null; cd %s; qsub %s/job.sh -t 0-%d"' % (self._path, self._path, n - 1))

        os.chdir(cwd)

