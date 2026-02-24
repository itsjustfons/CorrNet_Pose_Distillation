#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -L jobenv=singularity
#PJM -j

HOME=/home/z40575r/
USER=z40575r

module load singularity

singularity exec \
    --bind $HOME,/data/group1/${USER} \
        --nv /data/group1/${USER}/latest.sif \
            bash ../pjsub/training_run_coco.sh