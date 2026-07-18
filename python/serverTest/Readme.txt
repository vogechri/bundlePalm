BundlePalm server/client build and clustering comparison
========================================================

All commands below assume the repository is located at:

    ~/bundlePalm

The client is a Python program and is not compiled. The build creates:

1. python/libprocess_clusters.so
   The clustering library loaded by serverTest/clustering.py.

2. python/serverTest/build/zeromq_cpp_server_ex
   The C++ optimization server.

3. python/serverTest/build/generated/proto/test_pb2.py
   The generated Python protobuf module used by client_acc.py.


1. Install system dependencies
------------------------------

On Ubuntu 24.04 or WSL Ubuntu, run:

    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        protobuf-compiler \
        libprotobuf-dev \
        libzmq3-dev \
        cppzmq-dev \
        libceres-dev \
        libeigen3-dev \
        python3-dev \
        python3-pip \
        python3-venv

Verify the main build tools:

    g++ --version
    cmake --version
    protoc --version
    python3 --version


2. Create the Python environment
--------------------------------

Run these commands from python/serverTest:

    cd ~/bundlePalm/python/serverTest
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install numpy scipy pyzmq torch protobuf==3.20.3

Check the imports before building:

    python -c "import numpy, scipy, zmq, torch, google.protobuf; print('Python dependencies OK')"

The error

    ModuleNotFoundError: No module named 'zmq'

means that client_acc.py was run with a Python interpreter where pyzmq is not
installed. Activate .venv or invoke .venv/bin/python explicitly.


3. Build the clustering library and server
------------------------------------------

From python/serverTest, with or without the virtual environment active:

    cd ~/bundlePalm/python/serverTest
    ./build_local.sh

Expected output files:

    ../libprocess_clusters.so
    build/zeromq_cpp_server_ex
    build/generated/proto/test_pb2.py

Optional: test the clean landmark partitioner:

    cd ~/bundlePalm/python
    g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
        camera_hypergraph_partitioning.cpp \
        landmark_partitioning.cpp \
        test_landmark_partitioning.cpp \
        -o /tmp/test_landmark_partitioning
    /tmp/test_landmark_partitioning

The expected final line is:

    landmark partitioning tests passed

The separate camera_hypergraph_partitioning implementation is retained for a
different experiment but is not selected by client_acc.py.


4. Start the optimization server
--------------------------------

Use a dedicated terminal and leave the process running:

    cd ~/bundlePalm/python/serverTest
    ./build/zeromq_cpp_server_ex

The server listens on TCP ports 5555, 5556, and 5557. Stop it with Ctrl-C.

If startup reports that an address is already in use, an older server process
is probably still running. Stop that process before starting another server.


5. Run one baseline test
------------------------

Open a second terminal:

    cd ~/bundlePalm/python/serverTest
    source .venv/bin/activate
    export PYTHONPATH="$PWD/build/generated/proto${PYTHONPATH:+:$PYTHONPATH}"

Run the existing landmark partitioner:

    BUNDLE_PALM_CLUSTERING=landmark python -u client_acc.py \
        http://grail.cs.washington.edu/projects/bal/data/ladybug/ \
        problem-49-7776-pre.txt.bz2 30 10 \
        > baseline_landmark.log 2>&1

Arguments after client_acc.py are:

    BASE_URL  PROBLEM_FILE  OPTIMIZATION_ITERATIONS  NUMBER_OF_CLUSTERS

Use problem-49-7776 and 5 to 30 iterations for the first test. The original
run.sh starts many large experiments and should not be used for a smoke test.

client_acc.py is inside python/serverTest. Running this command from python:

    python -u client_acc.py ...

fails because python/client_acc.py does not exist. Either change to
python/serverTest as shown above or use the complete script path. Changing to
python/serverTest is preferred because clustering.py loads
../libprocess_clusters.so relative to the current directory.


6. Run the clean landmark partitioner
-------------------------------------

Keep the server running and execute:

    BUNDLE_PALM_CLUSTERING=landmark_clean python -u client_acc.py \
        http://grail.cs.washington.edu/projects/bal/data/ladybug/ \
        problem-49-7776-pre.txt.bz2 30 10 \
        > clean_landmark_partition.log 2>&1

The clean implementation assigns every landmark to exactly one cluster. All
observations of that landmark follow it. Cameras can occur in several clusters,
which matches the optimizer's existing camera-consensus workflow. Landmarks
with identical camera signatures are processed together during initialization
to preserve the strongest camera communities.

Residual balance slack defaults to 5 percent. To use 1 percent slack:

    BUNDLE_PALM_CLUSTERING=landmark_clean \
    BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 \
    BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 \
    python -u client_acc.py \
        http://grail.cs.washington.edu/projects/bal/data/ladybug/ \
        problem-49-7776-pre.txt.bz2 30 10 \
        > clean_landmark_partition_slack.log 2>&1

Valid slack values range from 0.0 through 0.05. Because landmarks are
indivisible weighted items, exact bounds may occasionally be impossible. The
reported residual balance violation is zero when every cluster is within the
requested bounds.

BUNDLE_PALM_MIN_CAMERA_LANDMARKS defaults to 20. Camera-cluster incidences with
1 through 19 landmarks are considered weak. Refinement first minimizes the
number of weak incidences, then moves each remaining weak incidence toward the
nearest valid state: absent from the cluster or supported by at least the
configured number of landmarks. Residual imbalance and camera copies are lower
priority objectives. When single-landmark moves cannot cross an intermediate
weak state, refinement evaluates the complete operation atomically: either all
landmarks of the weak camera are evacuated from that cluster, or enough
landmarks observed by that camera are imported to reach the support threshold.
If the receiving cluster lacks residual capacity, unrelated landmarks are
moved out within the same transaction. High-degree bridge landmarks can create
several secondary weak camera incidences at once; these are repaired as a
bounded closure before the transaction is evaluated. The complete transaction
is committed only when it improves the final objective and all clusters remain
within the residual balance bounds.

Clean landmark mode reports:

    - residuals, landmarks, and cameras per cluster
    - requested residual balance bounds and any violation
    - weak camera-cluster counts below the configured minimum support
    - additional camera copies across clusters
    - clustering time

To compare the optional camera-straggler objective against the baseline, run
the same command twice. The default and explicit baseline are equivalent:

    BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=0 ...

Enable minimization of the largest per-cluster camera count with:

    BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 ...

When enabled, maximum camera count is optimized after weak camera quality and
before total additional camera copies. Residual balance bounds remain hard.

The scalable implementation is selected with:

    BUNDLE_PALM_CLUSTERING=landmark_scalable ...

The stability-focused scalable implementation is selected with:

    BUNDLE_PALM_CLUSTERING=landmark_scalable_stable ...

It preserves the scalable algorithm and residual-balance constraints. It first
minimizes the total degree 1-9 camera incidences, then uses a steep finite
penalty to prefer safer compositions when that count is equal. Degrees 1 and 2
have additional penalties, while degrees 3-9 use the cubic deficit from degree
10. Additional camera copies and maximum cameras in a cluster follow, with
degree 10 through the configured weak-support limit last. The original
landmark_scalable objective remains available for comparisons.

It reports progress approximately every five seconds by default. Set
BUNDLE_PALM_PARTITION_TRACE=0 to disable progress or set
BUNDLE_PALM_PARTITION_TRACE_INTERVAL_SECONDS to change the interval. Each
run reports BAL archive validation and parsing first, followed by camera-seed
partitioning and scalable landmark partitioning. Partition reports include the
phase, recovery round and pass, work, accepted moves, threshold-relative severe
and weak incidences, the absolute degree 1 through 9 incidence count, camera
copies, maximum cameras in a cluster, and the current residual range.

Lowering BUNDLE_PALM_MIN_CAMERA_LANDMARKS relaxes the support objective; it does
not directly remove low-degree camera-cluster incidences. For example, 20
classifies degrees 1 through 9 as severe and 1 through 19 as weak, while 10
classifies degrees 1 through 4 as severe and 1 through 9 as weak. Compare the
absolute degree 1 through 9 statistic when evaluating different thresholds.


7. Run the trimmed automatic comparison
----------------------------------------

The comparison script runs the original landmark mode followed by the clean
landmark mode:

    cd ~/bundlePalm/python/serverTest
    ./run_compare_clustering.sh \
        http://grail.cs.washington.edu/projects/bal/data/ladybug/ \
        problem-49-7776-pre.txt.bz2 \
        30 \
        10

It automatically uses .venv/bin/python when that environment exists. It
creates:

    compare_landmark.log
    compare_landmark_clean.log

To inspect the main measurements manually:

    grep -E "clustering took|Residual balance|Weak camera-cluster|Additional camera copies|DRE BFGS|f\(v\)=" \
        compare_landmark.log compare_landmark_clean.log

For the most controlled comparison, restart the C++ server between the two
modes and run the commands from sections 5 and 6 separately. The automatic
script reuses one server process.


8. Interpreting the comparison
------------------------------

Both modes partition landmarks exclusively and copy cameras between clusters.
They therefore use the same optimizer variables and camera-consensus path.
Clustering time, worker runtime, convergence, and final cost are directly
comparable. Restarting the server before each mode provides the cleanest timing
comparison.


9. Common errors
----------------

Error:

    python: can't open file '.../python/client_acc.py'

Fix:

    cd ~/bundlePalm/python/serverTest

Error:

    ModuleNotFoundError: No module named 'zmq'

Fix:

    cd ~/bundlePalm/python/serverTest
    source .venv/bin/activate
    python -m pip install pyzmq

Error:

    ModuleNotFoundError: No module named 'test_pb2'

Fix: rebuild the server and export the generated module directory:

    ./build_local.sh
    export PYTHONPATH="$PWD/build/generated/proto${PYTHONPATH:+:$PYTHONPATH}"

Error:

    OSError: ../libprocess_clusters.so: cannot open shared object file

Fix: run ./build_local.sh and launch the client from python/serverTest.

Error:

    cmake is required to build the optimization server

Fix:

    sudo apt install cmake
