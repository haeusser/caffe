#!/usr/bin/python

import sys
from pymill.Toolbox import readPFM
import numpy as np 
import os
  
  
if len(sys.argv) < 3 or len(sys.argv) > 4:
  raise ValueError('Usage: make_FlyingStuff_clip_def DATASET SUBFOLDER (THRESHOLD)')
if len(sys.argv) >= 3:
  dataset = sys.argv[1]
  subfolder = sys.argv[2]
if len(sys.argv) == 4:
  threshold = int(sys.argv[2])
else:
  threshold = 500
  
min_clip_length = 10
  
print "Dataset %s, subfolder %s, threshold %d" % (dataset, subfolder, threshold) 

data_path = '/misc/lmbraid17/sceneflownet/common/data/2_blender-out/lowres/' + dataset + '/' + subfolder
clipdef_path = '/misc/lmbraid17/sceneflownet/common/data/2_blender-out/lowres/collection_def_' + dataset + '.txt'
subfolder_template = data_path + '/%.4d'
disp_L_template = subfolder_template + '/converted/Disparity_%.4d_L.pfm'
disp_R_template = subfolder_template + '/converted/Disparity_%.4d_R.pfm'

clipdef_file = open(clipdef_path, "a");
clipdef_file.write('\n# ===== ' + dataset + '/' + subfolder + ' collection def =====\n\n')         
clipdef_file.write('%-25s %-20s %-20s %-20s %-20s\n\n' % ('# movie', 'setting', 'start frame', 'end frame', 'collection name'))
line_template = '%-25s %-20s %-20.4d %-20.4d %-20s\n'

in_clip = False
for num_dir in range(1000):
  if os.path.isdir(subfolder_template % num_dir):
    print 'Processing ', subfolder_template % num_dir
    num_file = 1
    in_clip = False
    # this is in case the first frame does not have number 1
    while (not os.path.exists(disp_L_template % (num_dir, num_file))) and (num_file < 100):
      num_file += 1
    # found the first existing frame and start processing
    while os.path.exists(disp_L_template % (num_dir, num_file)):
      dispL,_ = readPFM(disp_L_template % (num_dir, num_file))
      dispR,_ = readPFM(disp_R_template % (num_dir, num_file))
      dispL = np.abs(dispL)
      dispR = np.abs(dispR)
      if (dispL > threshold).any() or (dispR > threshold).any():
        print 'Bad frame ', num_file, ' , max dispL ', np.max(dispL), ' , max dispR ', np.max(dispR)
        if in_clip:
          if num_file - start_frame + 1 >= min_clip_length:
            print 'Writing clip frames %d to %d' % (start_frame, num_file - 1)
            clipdef_file.write(line_template % (dataset, subfolder + '/' + '%.4d' % num_dir, start_frame, num_file - 1, subfolder))
          in_clip = False
      else:
        if not in_clip:
          start_frame = num_file
          in_clip = True    
      
      clipdef_file.flush()
      num_file += 1
    
  
  if in_clip:
    if num_file - start_frame + 1 >= min_clip_length:
      print 'Writing clip frames %d to %d' % (start_frame, num_file - 1)
      clipdef_file.write(line_template % (dataset, subfolder + '/' + '%.4d' % num_dir, start_frame, num_file - 1, subfolder))
      
  in_clip = False
  
  
clipdef_file.close()
    

