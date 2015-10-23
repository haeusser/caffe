#!/usr/bin/python

import os
import sys
import argparse
from pymill import Toolbox as tb
from pymill.SceneFlowNet import Dataset as ds

dataPath = '/misc/lmbraid17/sceneflownet/common/data/2_blender-out/lowres'

defs = []
for file in os.listdir(dataPath):
    if file.endswith('.txt'):
        print 'reading', file
        defs.append('%s/%s' % (dataPath, file))

collections = ds.readCollections(defs)

tb.pprint(collections)

def getSelectedCollections():
    collectionNames = collections.keys()

    selectedNames = []
    if args.collections == '':
        selectedNames = collectionNames
    else:
        exprs = args.collections.split(',')
        for expr in exprs:
            selectedNames += tb.wildcardMatch(collectionNames, expr)

    selectedNames = tb.unique(selectedNames)

    selectedCollections = {}
    for name in selectedNames:
        for collectionName, collection in collections.iteritems():
            if collectionName == name:
                selectedCollections[name] = collection

    return selectedCollections


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help='verbose', action='store_true')
subparsers = parser.add_subparsers(dest='command', prog='sfn-ds')

# create-bindb
subparser = subparsers.add_parser('create-bindb', help='create bindary db')
subparser.add_argument('--collections', help='list of collections', default = '')
subparser.add_argument('--downsample', help='downsampling factor', default = 1, type = int)
subparser.add_argument('--skip-if-exists', help='skip if database exists', action='store_true')

# create-histograms
subparser = subparsers.add_parser('create-histograms', help='compute histograms')
subparser.add_argument('--collections', help='list of collections', default = '')
subparser.add_argument('--skip-if-exists', help='skip if database exists', action='store_true')

# create-lmdb
subparser = subparsers.add_parser('create-lmdb', help='create LMDB database')
subparser.add_argument('rendertype', help='rendertype', choices=['clean', 'final'])
subparser.add_argument('type', help='data type', choices=['flow', 'disparity', 'sceneflow'])
subparser.add_argument('name', help='collection name')
subparser.add_argument('--collections', help='list of collections', default = '')
subparser.add_argument('--entity-size', help='entity size', default = 2, type=int)
subparser.add_argument('--downsample', help='downsampling factor', default = 1, type = int)
subparser.add_argument('--skip-if-exists', help='skip if database exists', action='store_true')

args = parser.parse_args()
tb.verbose = args.verbose

if args.command == 'create-bindb':
    selectedCollections = getSelectedCollections()

    for name, collection in selectedCollections.iteritems():
        print '%s: %s' % (name, collection)

        ds.createBinDB(
            resolution='960x540',
            subpath='.',
            collectionName=name,
            clips=collection,
            downsample=args.downsample,
            skipIfExists=args.skip_if_exists
        )


elif args.command == 'create-histograms':
    selectedCollections = getSelectedCollections()

    for name, collection in selectedCollections.iteritems():
        print '%s: %s' % (name, collection)

        ds.computeHistograms(
            resolution='960x540',
            subpath='.',
            collectionName=name,
            clips=collection,
            skipIfExists=args.skip_if_exists
        )

elif args.command == 'create-lmdb':
    selectedCollections = getSelectedCollections()

    clips = []
    for name, collection in selectedCollections.iteritems():
        print '-----------))))))))))))))))))))))))))))))))))'
        print collection
        clips += collection


    # ds.createLMDB(
    #     rendertype=args.rendertype,
    #     type=args.type,
    #     name=args.name,
    #     clips=clips,
    #     entitySize=args.entity_size,
    #     downsample=args.downsample,
    #     skipIfExists=args.skip_if_exists
    # )

    print clips













# # check-flow
# subparser = subparsers.add_parser('check-flow', help='compute histograms')
# subparser.add_argument('--collections', help='list of collections', default = '')

#
# elif args.command == 'check-flow':
#     rendertype = 'clean'
#
#     clips = getClips(resolution)
#
#     print '%s:' % resolution
#     for clip in clips:
#         print clip
#
#         ds.makeFlowCheck(
#             resolution=resolution,
#             rendertype=rendertype,
#             clip=clip
#         )



# # create-lmdb
# subparser = subparsers.add_parser('create-lmdb', help='create LMDB database')
# subparser.add_argument('resolution', help='resolution', choices=['lowres', 'highres'])
# subparser.add_argument('rendertype', help='rendertype', choices=['clean', 'final'])
# subparser.add_argument('type', help='data type', choices=['flow', 'disparity', 'sceneflow'])
# subparser.add_argument('name', help='dataset name')
# subparser.add_argument('--collections', help='list of collections', default = '')
# subparser.add_argument('--entity-size', help='entity size', default = 2, type=int)
# subparser.add_argument('--downsample', help='downsampling factor', default = 1, type = int)
# subparser.add_argument('--skip-if-exists', help='skip if database exists', action='store_true')
#
# elif args.command == 'create-lmdb':
#     resolution = args.resolution
#     rendertype = args.rendertype
#
#     clips = getClips(resolution)
#
#     print '%s clips:' % resolution
#     for clip in clips:
#          print clip
#
#     ds.createLMDB(
#         resolution=resolution,
#         rendertype=args.rendertype,
#         type=args.type,
#         name=args.name,
#         clipList=[clip for clip in clipCollections[resolution] if clip.name() in clips],
#         entitySize=args.entity_size,
#         downsample=args.downsample,
#         skipIfExists=args.skip_if_exists
#     )
