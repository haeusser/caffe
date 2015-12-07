#!/usr/bin/python

import signal
from pymill import OpticalFlow
import os
import sys
from string import Template
from termcolor import colored
import argparse
import numpy as np
import pylab as plt
import commands
import multiprocessing
import subprocess
import atexit
import shutil
from pymill import Toolbox as tb

class PyFlow:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--format', help='id formatting', default='%04d')
        parser.add_argument('--verbose', help='verbose', action='store_true')
        parser.add_argument('--label', help='restrict to label', type=int, default=-1)
        parser.add_argument('--limit', help='limit number of entities', type=str, default='')
        parser.add_argument('--id', help='restrict to id', type=int)

        subparsers = parser.add_subparsers(dest='command')

        # datasets
        datasets_parser = subparsers.add_parser('datasets')

        # uents
        uents_parser = subparsers.add_parser('uents')
        uents_parser.add_argument('datasets', help='datasets')

        # bents
        bents_parser = subparsers.add_parser('bents')
        bents_parser.add_argument('datasets', help='datasets')

        # compute
        compute_parser = subparsers.add_parser('compute')
        compute_parser.add_argument('methods', help='method definitions')
        compute_parser.add_argument('datasets', help='datasets')
        compute_parser.add_argument('--local', help='do not compute on cluster', action='store_true')
        compute_parser.add_argument('--cores', help='run on multiple cores', default=1, type=int)

        # update
        update_parser = subparsers.add_parser('update')
        update_parser.add_argument('methods', help='method definitions')
        update_parser.add_argument('datasets', help='datasets')
        update_parser.add_argument('--local', help='do not compute on cluster', action='store_true')
        update_parser.add_argument('--cores', help='run on multiple cores', default=1, type=int)

        # check
        check_parser = subparsers.add_parser('check')
        check_parser.add_argument('methods', help='method definition')
        check_parser.add_argument('datasets', help='datasets')

        # clean
        clean_parser = subparsers.add_parser('clean')
        clean_parser.add_argument('methods', help='method definition')
        clean_parser.add_argument('datasets', help='datasets')

        #
        # integrate_parser = subparsers.add_parser('integrate')
        # integrate_parser.add_argument('method', help='method definition')
        # integrate_parser.add_argument('--inext', help='input extension', default='')
        # integrate_parser.add_argument('--outext', help='output extension', default='')
        #
        # # integrate
        # integrate_parser = subparsers.add_parser('integrate')
        # integrate_parser.add_argument('method', help='method definition')
        # integrate_parser.add_argument('--inext', help='input extension', default='')
        # integrate_parser.add_argument('--outext', help='output extension', default='')
        #

        # epe
        epe_parser = subparsers.add_parser('epe')
        epe_parser.add_argument('methods', help='method definition')
        epe_parser.add_argument('datasets', help='datasets')
        epe_parser.add_argument('--type', help='type of epe', default='all')
        epe_parser.add_argument('--make-epe', help='compute epe', action='store_true')
        epe_parser.add_argument('--make-stat', help='compute stat', action='store_true')
        epe_parser.add_argument('--refresh', help='refresh', action='store_true')
        epe_parser.add_argument('--no-output', help='no output', action='store_true')
        epe_parser.add_argument('--list', help='list errors', action='store_true')
        epe_parser.add_argument('--plain', help='plain output', action='store_true')

        # epe-stat
        epe_stat_parser = subparsers.add_parser('epe-stat')
        epe_stat_parser.add_argument('methods', help='method definition')
        epe_stat_parser.add_argument('datasets', help='datasets')
        epe_stat_parser.add_argument('--type', help='type of epe', default='all')
        epe_stat_parser.add_argument('--make-epe', help='compute epe', action='store_true')
        epe_stat_parser.add_argument('--make-stat', help='compute stat', action='store_true')
        epe_stat_parser.add_argument('--refresh', help='refresh', action='store_true')

        # plot
        plot_parser = subparsers.add_parser('plot')
        plot_parser.add_argument('methods', help='method definition')
        plot_parser.add_argument('datasets', help='datasets')
        plot_parser.add_argument('colordef', help='color definition',default='', nargs='?')
        plot_parser.add_argument('--type', help='type of epe', default='all')
        plot_parser.add_argument('--diff', help='plot difference', action='store_true')

        self._args = parser.parse_args()
        OpticalFlow.UnaryEntity.format = self._args.format
        OpticalFlow.BinaryEntity.format = self._args.format
        tb.verbose = self._args.verbose

        self._datasets = []
        if 'datasets' in self._args:
            self._datasets = OpticalFlow.expandDataset(self._args.datasets, self._args.label, self._args.limit)

        self._methods = []
        if 'methods' in self._args:
            self._methods = OpticalFlow.Specification.expand(self._args.methods)

    def datasets(self):
        for name in OpticalFlow.Dataset.names():
            print name

    def uents(self):
        for ds in self._datasets:
            for uent in ds.uents():
                print uent

    def bents(self):
        for ds in self._datasets:
            print ds
            for bent in ds.bents():
                print bent

    def compute(self):
        queue = tb.Queue()

        for m in self._methods:
            for ds in self._datasets:
                tb.notice('creating jobs for <%s> on <%s>' % (m, ds))
                for ent in ds.uents() if m.direction() == '' else ds.bents():
                    job = tb.Job()
                    ent.bind(m).makeComputeJob(job)
                    queue.postJob(job)

            queue.finishPacket()

        queue.submit(local=self._args.local, cores=self._args.cores)

    def update(self):
        queue = tb.Queue()

        for m in self._methods:
            for ds in self._datasets:
                tb.notice('creating jobs for <%s> on <%s>' % (m, ds))
                for ent in ds.uents() if m.direction() == '' else ds.bents():
                    job = tb.Job()
                    ent.bind(m).makeUpdateJob(job)
                    queue.postJob(job)

            queue.finishPacket()

        queue.submit(local=self._args.local, cores=self._args.cores)

    def check(self):
        nTotal = 0
        nOk = 0
        for m in self._methods:
            for ds in self._datasets:
                for ent in ds.uents() if m.direction() == '' else ds.bents():
                    if ent.bind(m).checkOut(self._args.verbose):
                        nOk += 1
                    nTotal += 1

        if nOk == nTotal:
            tb.notice('(%d/%d) passed' % (nOk, nTotal), 'passed')
        else:
            tb.notice('(%d/%d) passed' % (nOk, nTotal), 'failed')

    def clean(self):
        nCleaned = 0
        for m in self._methods:
            for ds in self._datasets:
                for ent in ds.bents():
                    if ent.bind(m).clean():
                        nCleaned += 1

        tb.notice('cleaned %d entries' % (nCleaned), 'passed')

    def epe(self):
        epe = {}
        for m in self._methods:
            entries = []
            for ds in self._datasets:
                for ent in ds.bents():
                    entries.append(ent.bind(m).flowStatParams())

            cmd = 'FlowStat --epe-type=%s %s %s %s' % (
                       self._args.type,
                       '--make-epe' if self._args.make_epe else '',
                       '--make-stat' if self._args.make_stat else '',
                       '--refresh' if self._args.refresh else ''
            )

            epe[str(m)] = [line for line in tb.run(cmd, '\n'.join(entries)).split('\n') if line.strip() != '']

        if self._args.no_output:
            return

        fw = [0]
        for m in epe: fw.append(max(len(m), 10))
        for ds in self._datasets:
            for ent in ds.bents():
                l = len(ent.detailedName())
                if l > fw[0]:
                    fw[0] = l

        s = 0
        for n in fw: s += n
        s += len(fw) - 1

        if self._args.list:
            if self._args.plain:
                i = 0
                for ds in self._datasets:
                    for ent in ds.bents():
                        for list in epe.itervalues():
                            print list[i],
                        print
                        i += 1

                for list in epe.itervalues():
                    print list[-1],
                print
            else:
                print ' ' * fw[0],
                j = 1
                for m in epe.iterkeys():
                    print '{0:>{1}s}'.format(m, fw[j]),
                    j += 1
                print

                i = 0
                print '-' * s
                for ds in self._datasets:
                    for ent in ds.bents():
                        print '{0:{1}s}'.format(ent.detailedName(), fw[0]),
                        j = 1
                        for list in epe.itervalues():
                            print '{0:{1}s}'.format(list[i], fw[j]),
                            j += 1
                        print
                        i += 1
                print '-' * s

                print '{0:{1}s}'.format('average', fw[0]),
                j = 1
                for list in epe.itervalues():
                    print '{0:>{1}s}'.format(list[-1].replace('avg=', '').strip(), fw[j]),
                    j += 1
                print
        else:
            if len(entries) > 1:
                maxMLen = 0
                for i in range(1,len(fw)):
                    if fw[i] > maxMLen: maxMLen = fw[i]

                for m, list in epe.iteritems():
                    print '{0:{1}s}'.format(m, maxMLen),
                    if not len(list):
                        print
                        continue
                    print '{0:>{1}s}'.format(list[-1].replace('avg=', '').strip(), 10),
                    print
            else:
                for list in epe.itervalues():
                    print list[-1].replace('avg=','').strip()

    def epeStat(self):
        if len(self._methods) > 1:
            raise Exception('cannot make epe stat for more than one method')

        method = self._methods[0]
        entries = []
        for ds in self._datasets:
            for ent in ds.bents():
                entries.append(ent.bind(method).flowStatParams())

        cmd = 'FlowStat --epe-type=%s %s %s %s %s' % (
                   self._args.type,
                   '--make-epe' if self._args.make_epe else '',
                   '--make-stat' if self._args.make_stat else '',
                   '--refresh' if self._args.refresh else '',
                   '--stat'
        )

        print tb.run(cmd, '\n'.join(entries)),

    def plot(self):
        for ds in self._datasets:
            plot = OpticalFlow.EpePlot(ds)
            plot.addMethods(self._methods,self._args.type)
            if self._args.diff: plot.makeDiff()
            plot.setColorDef(self._args.colordef)
            plot.plot()

        plt.show()

    def run(self):
        if   self._args.command == 'datasets':              self.datasets()
        elif self._args.command == 'uents':                 self.uents()
        elif self._args.command == 'bents':                 self.bents()
        elif self._args.command == 'compute':               self.compute()
        elif self._args.command == 'update':                self.update()
        elif self._args.command == 'check':                 self.check()
        elif self._args.command == 'clean':                 self.clean()
        elif self._args.command == 'epe':                   self.epe()
        elif self._args.command == 'epe-stat':              self.epeStat()
        elif self._args.command == 'plot':                  self.plot()


        #
        #     if self._args.subCommand == 'list':             self.datasetList()
        #     elif self._args.subCommand == 'uents':          self.datasetUEnts()
        #     elif self._args.subCommand == 'bents':          self.datasetBEnts()
        #     else: print 'no sub command'
        # else:
        #     if self._args.command == 'compute':             self.compute()
        #     if self._args.command == 'check':               self.check()
        #     if self._args.command == 'update':              self.update()
        #     if self._args.command == 'epe':                 self.epe()
        #     if self._args.command == 'plot':                self.plot()


if __name__ == '__main__':
    PyFlow().run()


    # def localIntegrate(self):
    #     method = self._args.method
    #     for ent in self._ents:
    #         ent.integrateLocal(method,self._args.inext)
    #
    #
    # def update(self):
    #     queue = tb.Queue()
    #     for ent in self._ents:
    #         method = OpticalFlow.method(ent,self._args.method)
    #         if not method.checkOut(False):
    #             method.compute(queue)
    #
    #     queue.submit(here=self._args.here,cores=self._args.cores)
    #
    # def plot(self):
    #     defs = self._args.method.split(':')
    #     plot = OpticalFlow.Plot(self._ds)
    #
    #     for d in defs:
    #         print 'listing epe for %s'%(d)
    #
    #         data=self._ds.epeList(d)
    #         plot.addEpeValues(d,data,{});
    #
    #     plot.setColorDef(self._args.colordef)
    #     plot.plot()
    #     plt.show()
    #

        # if self._args.command == 'local':
        #     if self._args.localCommand == 'epe':            self.localEpe()
        #     elif self._args.localCommand == 'refine':       self.localRefine()
        #     elif self._args.localCommand == 'integrate':    self.localIntegrate()



        # dataset
        # if self._args.dataset!='':
        #     self._ds = OpticalFlow.Dataset(self._args.dataset)
        # else:
        #     self._ds = OpticalFlow.Dataset.infer()

        # self._ents = []
        #
        # # ids
        # self._ids = []
        # if self._args.id is not None:
        #     self._ids = [self._args.id]
        # elif self._args.label is not None:
        #     self._ids = self._ds.idsByLabel(self._args.label)
        # else:
        #     self._ids = self._ds.ids()
        #
        # self._ents = self._ds.entsForIds(self._ids)



































        # # local epe
        # local_epe_parser = local_subparsers.add_parser('epe')
        # local_epe_parser.add_argument('--inext', help='input extension', default='')
        # local_epe_parser.add_argument('--outext', help='output extension', default='')
        # local_epe_parser.add_argument('--type', help='type of epe', default='all')
        # local_epe_parser.add_argument('--make-epe', help='compute epe', action='store_true')
        # local_epe_parser.add_argument('--make-stat', help='compute stat', action='store_true')
        # local_epe_parser.add_argument('--no-output', help='no output', action='store_true')

                #
                #
                # elif self._args.localSubcommand == 'list':
                #     map = []
                #     for ent in self._ents:
                #         map.append(ent.formattedId())
                #
                #     cmd = 'FlowStat %s --files=%s --epe-type=%s --def=%s' % (
                #                self._ds.name(),
                #                ','.join(map),
                #                self._args.epe_type,
                #                self._args.method
                #     )
                #
                #     epe=Toolbox.run(cmd)
                #
                #     print epe
                #


                # terminating = multiprocessing.Event()
                #
                # print self._args.cores
                # pool = multiprocessing.Pool(processes=self._args.cores)
                # pool.map_async(runLowresVarRefinement,self._ents).get(999999)
                # pool.close()




# pyflow=None
#
# def runLowresVarRefinement(ent):
#     temp = Template("VarFlowRefine $img1 $img2 $flo $out $boundaries $params");
#
#     outExt=pyflow._args.outext
#     if outExt=='':
#         outExt='.refined.flo'
#     else:
#         if outExt[0]!='.':
#             outExt='.'+outExt
#
#     if outExt[-4:]!='.flo':
#         outExt+=".flo"
#
#     flowFile=ent.idLowresFlo(pyflow._args.inext);
#     outFile=flowFile.replace('.flo',outExt)
#
#     boundaries=''
#     if pyflow._args.boundaries:
#         boundaries='--boundaries='+ent.boundaryPath()
#
#     command = temp.safe_substitute(
#         img1=ent.img1Path(),
#         img2=ent.img2Path(),
#         flo=flowFile,
#         out=outFile,
#         boundaries=boundaries,
#         params=pyflow._args.refine_params
#     )
#
#     print command
#     p=subprocess.Popen(['/bin/bash','-c',command])
#     p.wait()
#
# def runCompute(ent):
#     ent.chdirParent()
#
#     if pyflow._args.missing:
#         path = ent.path()+"/"+pyflow._args.method+"/flow.flo"
#         if os.path.isfile(path):
#             print "skipping",path
#             return
#
#     temp = Template("flow compute $method $file1,$file2 --verbose")
#     command = temp.safe_substitute(
#         file1=ent.img1Filename(),
#         file2=ent.img2Filename(),
#         method=pyflow._args.method
#     )
#
#     print "processing %s, %s" % (ent.img1Path(), command)
#
#     p=subprocess.Popen(['/bin/bash','-c',command])
#     p.wait()
