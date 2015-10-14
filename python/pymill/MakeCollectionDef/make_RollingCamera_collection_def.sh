rm /misc/lmbraid17/sceneflownet/common/data/2_blender-out/960x540/collection_def_FlyingThings3D_RollingCamera.txt
python make_FlyingStuff_clip_def.py FlyingThings3D_RollingCamera TEST/A | tee collection_def_RollingCamera_log.txt
python make_FlyingStuff_clip_def.py FlyingThings3D_RollingCamera TEST/B | tee -a collection_def_RollingCamera_log.txt
python make_FlyingStuff_clip_def.py FlyingThings3D_RollingCamera TRAIN/A | tee -a collection_def_RollingCamera_log.txt
python make_FlyingStuff_clip_def.py FlyingThings3D_RollingCamera TRAIN/B | tee -a collection_def_RollingCamera_log.txt

