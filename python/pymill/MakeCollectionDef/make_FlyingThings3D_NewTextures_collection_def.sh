rm /misc/lmbraid17/sceneflownet/common/data/2_blender-out/960x540/collection_def_FlyingThings3D_NewTextures.txt
python make_collection_def.py FlyingThings3D_NewTextures TEST/A | tee collection_def_FlyingThings3D_NewTextures_log.txt
python make_collection_def.py FlyingThings3D_NewTextures TEST/B | tee -a collection_def_FlyingThings3D_NewTextures_log.txt
python make_collection_def.py FlyingThings3D_NewTextures TRAIN/A | tee -a collection_def_FlyingThings3D_NewTextures_log.txt
python make_collection_def.py FlyingThings3D_NewTextures TRAIN/B | tee -a collection_def_FlyingThings3D_NewTextures_log.txt

