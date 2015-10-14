rm /misc/lmbraid17/sceneflownet/common/data/2_blender-out/960x540/collection_def_StaticThings3D.txt
python make_FlyingStuff_clip_def.py StaticThings3D TEST/A | tee collection_def_StaticThings_log.txt
python make_FlyingStuff_clip_def.py StaticThings3D TEST/B | tee -a collection_def_StaticThings_log.txt
python make_FlyingStuff_clip_def.py StaticThings3D TRAIN/A | tee -a collection_def_StaticThings_log.txt
python make_FlyingStuff_clip_def.py StaticThings3D TRAIN/B | tee -a collection_def_StaticThings_log.txt

