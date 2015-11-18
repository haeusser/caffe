#!/bin/bash
echo 'STARTING DASHBOARD REFRESH SCRIPT'
sleep 1
whoami 
ssh -i /home/haeusserp/.ssh/id_rsa haeusserp@frieda "cd /misc/lmbraid17/sceneflownet/haeusserp/hackathon-caffe2; echo '  pulling git ...'; git pull; echo '  killing current dashboard ...'; screen -X -S dashboard quit; echo '  migrating and starting new dashboard ...'; screen -S dashboard -dm bash -c 'cd python/pymill/dashboard/; ./migrate.sh; ./run.sh;'; echo 'DONE.';"
