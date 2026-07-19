# bundlePalm

partitioning cameras, landmarks go to BOTH parts (middle of edge = lm) edges are 2 residuals.

1) 1/2 r_i for DRS?
2) best fill up as what i had before for PALM
3) double check: f(x1,..,xi,..xn) -> wrt xi.
4) drs sum_fi()
is different significantly
fi(x1,x2) -> deriv wrt to both.
1/2 fi(x1,x2) -> still to both .. 
1/2(fi(const, x2) + fi(x1,const)) 
gain to split landmark could be to get FULL update locally per lm, problem need to add cams as well

## Optional BAE GPU backend

The optional backend in `python/bae_local_solver.py` uses
[pypose/bae](https://github.com/pypose/bae) for raw angle-axis Jacobian assembly
and CUDA/Warp block-sparse Schur operations. The default `cpu` solver does not
import or require BAE.

Both backends use the same damping, historical Nesterov recurrence, stopping
criterion, and nonlinear acceptance rule. PALM partition ownership,
sequential/frozen-anchor semantics, and outer acceleration remain in
`python/palm_ba.py`.

The experimental `schur_pcg_nesterov.py` used during development is currently an
untracked local file in a BAE checkout. It is not installed by the upstream BAE
package. The bundlePalm backend should reimplement only the required Nesterov
Schur operator against BAE's public sparse helpers instead of importing that
example.

### Installation

Use a dedicated environment until the BAE extension build and bundlePalm use one
verified PyTorch/CUDA combination. The following reproduces the currently tested
package versions on CUDA 12.x:

```bash
python3.12 -m venv .venv-bae
source .venv-bae/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.6.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install pypose==0.9.5 warp-lang==1.15.0 scipy numpy

git clone https://github.com/pypose/bae.git ../bae
git -C ../bae checkout cde4c06df713422be36e807e3ee2a5db756a657b
USE_CUDSS=0 python -m pip install --no-build-isolation -v -e ../bae
```

`USE_CUDSS=0` is sufficient for the planned Warp Schur/Nesterov backend. To build
BAE with cuDSS, follow the version-specific instructions in the BAE README.

Verify the installation before running bundle adjustment:

```bash
python - <<'PY'
import torch
import warp as wp
import bae

assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(wp.get_devices())
PY
```

The verified development machine currently uses BAE `0.2.4`, PyPose `0.9.5`,
Warp `1.15.0`, PyTorch `2.6.0+cu124`, and an RTX 4080 with 16 GiB memory.

Run one serialized frozen-anchor GPU experiment from the `python` directory:

```bash
/path/to/bae/.venv/bin/python -u palm_ba.py \
	http://grail.cs.washington.edu/projects/bal/data/ladybug/ \
	problem-49-7776-pre.txt.bz2 30 6 \
	--execution parallel --workers 1 --accelerator heavy_ball \
	--local-solver bae --device cuda:0
```

The BAE backend requires one worker. Sequential execution and parallel execution
with one worker retain their respective immediate-commit and frozen-anchor
semantics.

### Integration stages

1. Completed: raw 9-parameter BAL residuals, structured `2x9` camera and `2x3`
	landmark Jacobians, Schur assembly, Nesterov solve, and back-substitution run
	on CUDA.
2. Completed: a six-block Ladybug frozen-anchor epoch matched every CPU block
	gain and the final cost (`137615.7099`).
3. Completed: BAE residual models and static block topology are cached. Damping
	must remain vectorized across the diagonal BSR values; indexing one landmark
	block at a time forces a CUDA synchronization per landmark and dominates the
	entire solve.
4. On `problem-52-64053-pre`, one 30-block frozen-anchor epoch takes `2.4 s` of
	GPU solver time versus `6.8 s` on CPU. Both reach `9248310.324`; camera and
	landmark relative state errors are `2.92e-13` and `5.87e-15`. A warm full
	one-block local solve takes approximately `45-70 ms` on GPU.
5. Next: reuse sparse workspaces and benchmark longer sequential and
	frozen-anchor runs with outer acceleration.

CUDA candidate-cost evaluation and ownership-aware endpoint transfers are
required for these timings. A warm full-problem angle-axis step takes
approximately `45-70 ms`: about `9 ms` for the structured Jacobian,
`12 ms` for sparse assembly/conversion, `12 ms` for Nesterov, and the remainder
for state upload, candidate evaluation, and bookkeeping. Do not reintroduce CPU
residual evaluation in the BAE acceptance loop; a Venice NumPy objective pass
takes roughly `70-80 ms` by itself.

The standalone `examples/schur_pcg_nesterov.py` is an LM experiment, not the
default PALM policy. For a closer one-block comparison, use its inner stopping,
initial damping, rejection growth, and decrease-only acceptance policy:

```bash
/path/to/bae/.venv/bin/python -u palm_ba.py \
	. problem-52-64053-pre.txt.bz2 20 1 \
	--execution sequential --workers 1 --local-solver bae \
	--accelerator none --local-nfev 2 \
	--inner-iterations 400 --inner-check-interval 10 \
	--initial-damping 1e-6 --damping-reject-multiplier 16 \
	--local-acceptance decrease --runtime-weight 0 \
	--output palm52_gpu_lm_comparison.jsonl
```

On the verified machine this reaches `489028.02` in `3.0 s` for 20 accepted
angle-axis steps. The quaternion example reaches `479209.43` in `2.12 s`.
Steady-state step times are comparable; the remaining wall-time difference is
primarily first-step compilation and four initial angle-axis damping attempts
versus two quaternion attempts. The defaults remain the stricter PALM policy:
initial damping `1`, rejection multiplier `4`, per-iteration stopping checks,
and proximal-model acceptance.

### Future quaternion formulation

A separate PALM/DRS formulation may store rotations as unit quaternions and
solve each rotation block as a constrained least-squares problem. The unit-norm
constraint can lead to a small eigenvalue/SVD subproblem, depending on the local
quadratic model. This is different from BAE's current SE(3) tangent update and
should be evaluated as a separate algorithmic variant.

For that formulation, preconditioning must respect the constraint. Do not apply
independent diagonal scaling to the four ambient quaternion coefficients and
then renormalize. Either precondition the three-dimensional rotation tangent
system, or transform the constrained quadratic consistently and solve the
resulting generalized eigenvalue/SVD problem with its transformed norm
constraint. Outer heavy-ball, Anderson, and BFGS histories would likewise need
manifold-aware differences/retractions or a fixed tangent representation.

### Verified GPU reference

On `problem-52-64053-pre` the local experimental BAE example starts from the
same sum-of-squares objective as bundlePalm (`22304125.55`) and reaches
`479209.43` after 20 full-problem LM iterations in `2.47 s`, with approximately
431 MiB peak CUDA memory. Its configuration is 400 Nesterov Schur iterations,
inner tolerance `1e-2`, no local Jacobi preconditioner, and no scene
normalization. This is a full-problem LM reference, not a distributed PALM
result, but it demonstrates that end-to-end GPU Jacobian and Schur assembly is
the important performance target.