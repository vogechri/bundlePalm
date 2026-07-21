#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
DRS_SCALING=${DRS_SCALING:-jacobi_gmean}
export BUNDLE_PALM_DRS_SCALING="$DRS_SCALING"
echo "DRS camera scaling: $DRS_SCALING"

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-394-100368-pre.txt.bz2 90 30 2> lm394_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1064-113655-pre.txt.bz2 90 30 2> lm1064_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-245-198739-pre.txt.bz2 90 30 2> lm245_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1723-156502-pre.txt.bz2 90 30 2> lm1723_drs.txt #@ slow
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1266-132593-pre.txt.bz2 90 30 2> lm1266_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-931-102699-pre.txt.bz2 90 30 2> lm931_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-783-84444-pre.txt.bz2 90 30 2> lm783_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-89-110973-pre.txt.bz2 90 30 2> lm89_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-287-182023-pre.txt.bz2 90 30 2> lm287_drs.txt

# # standard run:
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-142-93602-pre.txt.bz2 90 30 2> lm142_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-646-73584-pre.txt.bz2 90 30 2> lm646_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-135-90642-pre.txt.bz2 90 30 2> lm135_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-52-64053-pre.txt.bz2 90 30 2> lm52_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-173-111908-pre.txt.bz2 90 30 2> lm173_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-356-226730-pre.txt.bz2 90 30 2> lm356_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-88-64298-pre.txt.bz2 90 30 2> lm88_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-49-7776-pre.txt.bz2 90 30 2> lm49_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/trafalgar/ problem-126-40037-pre.txt.bz2 90 30 2> lm126_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-427-310384-pre.txt.bz2 90 30 2> lm427_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-253-163691-pre.txt.bz2 90 30 2> lm253_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-961-187103-pre.txt.bz2 90 30 2> lm961_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/trafalgar/ problem-257-65132-pre.txt.bz2 90 30 2> lm257_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-744-543562-pre.txt.bz2 90 30 2> lm744_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-951-708276-pre.txt.bz2 90 30 2> lm951_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-308-195089-pre.txt.bz2 90 30 2> lm308_drs.txt
#Exhaustive repairs: slower, but preserves the quality-oriented search.
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-871-527480-pre.txt.bz2 90 30 2> lm871_drs.txt

# run:
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-1778-993923-pre.txt.bz2 90 30 2> lm1778_drs.txt
BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-1490-935273-pre.txt.bz2 90 30 2> lm1490_drs.txt

BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_MAX_REPAIR_WORK_PER_PHASE=500000000 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-3068-310854-pre.txt.bz2 90 30 2> lm3068_drs.txt # slow but main problem

# BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 BUNDLE_PALM_MAX_REPAIR_WORK_PER_PHASE=500000000 BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 BUNDLE_PALM_CLUSTERING=landmark_scalable BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-3068-310854-pre.txt.bz2 90 30 2> >(tee lm3068_drs_test.txt >&2)


# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-3068-310854-pre.txt.bz2 90 30 2> lm3068_drs.txt # slow but main problem
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-394-100368-pre.txt.bz2 90 30 2> lm394_drs.txt

# # # BASELINE ##########################
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1064-113655-pre.txt.bz2 90 30 2> lm1064_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-245-198739-pre.txt.bz2 90 30 2> lm245_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1723-156502-pre.txt.bz2 90 30 2> lm1723_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1266-132593-pre.txt.bz2 90 30 2> lm1266_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-931-102699-pre.txt.bz2 90 30 2> lm931_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-783-84444-pre.txt.bz2 90 30 2> lm783_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-3068-310854-pre.txt.bz2 90 30 2> lm3068_drs.txt

# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-89-110973-pre.txt.bz2 90 30 2> lm89_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-287-182023-pre.txt.bz2 90 30 2> lm287_drs.txt

# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-871-527480-pre.txt.bz2 90 30 2> lm871_drs.txt # slow

# # # # standard run:
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-142-93602-pre.txt.bz2 90 30 2> lm142_drs.txt
# # # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-394-100368-pre.txt.bz2 90 30 2> lm394_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-646-73584-pre.txt.bz2 90 30 2> lm646_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-135-90642-pre.txt.bz2 90 30 2> lm135_drs.txt

# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-52-64053-pre.txt.bz2 90 30 2> lm52_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-173-111908-pre.txt.bz2 90 30 2> lm173_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-931-102699-pre.txt.bz2 90 30 2> lm931_drs.txt

# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-356-226730-pre.txt.bz2 90 30 2> lm356_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-253-163691-pre.txt.bz2 90 30 2> lm253_drs.txt
# # # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1266-132593-pre.txt.bz2 90 30 2> lm1266_drs.txt

# # # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-89-110973-pre.txt.bz2 90 30 2> lm89_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-88-64298-pre.txt.bz2 90 30 2> lm88_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-49-7776-pre.txt.bz2 90 30 2> lm49_drs.txt
# # # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-245-198739-pre.txt.bz2 90 30 2> lm245_drs.txt

# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/trafalgar/ problem-126-40037-pre.txt.bz2 90 30 2> lm126_drs.txt
# # python -u client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-427-310384-pre.txt.bz2 90 30 2> lm427_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-253-163691-pre.txt.bz2 90 30 2> lm253_drs.txt
# # # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-931-102699-pre.txt.bz2 90 30 2> lm931_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-961-187103-pre.txt.bz2 90 30 2> lm961_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-1490-935273-pre.txt.bz2 90 30 2> lm1490_drs.txt

# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/trafalgar/ problem-257-65132-pre.txt.bz2 90 30 2> lm257_drs.txt
# # # # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-93-61203-pre.txt.bz2 90 30 2> lm93_drs.txt

# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-287-182023-pre.txt.bz2 90 30 2> lm287_drs.txt
# # # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-783-84444-pre.txt.bz2 90 30 2> lm783_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1064-113655-pre.txt.bz2 90 30 2> lm1064_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-744-543562-pre.txt.bz2 90 30 2> lm744_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-951-708276-pre.txt.bz2 90 30 2> lm951_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-871-527480-pre.txt.bz2 90 30 2> lm871_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-308-195089-pre.txt.bz2 90 30 2> lm308_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1723-156502-pre.txt.bz2 90 30 2> lm1723_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-1778-993923-pre.txt.bz2 90 30 2> lm1778_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-3068-310854-pre.txt.bz2 90 30 2> lm3068_drs.txt
# # #python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-13682-4456117-pre.txt.bz2 90 30 2> lm13682_drs.txt
# # python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-871-527480-pre.txt.bz2 90 30 2> lm871_drs.txt # slow
# # # 646 & 1266 fail at start (then ok), 931, 1064, 961, 3068 also not so good
##########################

# python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1064-113655-pre.txt.bz2 90 30 2> lm1064_drs.txt
# python client_acc.py http://grail.cs.washington.edu/projects/bal/data/venice/ problem-245-198739-pre.txt.bz2 90 30 2> lm245_drs.txt
# python client_acc.py http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1723-156502-pre.txt.bz2 90 30 2> lm1723_drs.txt
# python client_acc.py http://grail.cs.washington.edu/projects/bal/data/final/ problem-3068-310854-pre.txt.bz2 90 30 2> lm3068_drs.txt
