# BAE Schur experiment overlay

This directory preserves local experiment files using their paths relative to a
BAE checkout. It targets BAE revision
`cde4c06df713422be36e807e3ee2a5db756a657b` from
<https://github.com/pypose/bae>.

## Restore on another machine

Clone both repositories next to each other, check out the pinned BAE revision,
and copy the overlay onto BAE:

```bash
git clone https://github.com/vogechri/bundlePalm.git
git clone https://github.com/pypose/bae.git
git -C bae checkout cde4c06df713422be36e807e3ee2a5db756a657b
cp -a bundlePalm/python/bae_overlay/bae/. bae/bae/
cp -a bundlePalm/python/bae_overlay/examples/. bae/examples/
```

Create the BAE environment according to the main bundlePalm README. The BAL
archives are not part of this overlay; pass their directory with `--cache-dir`.

Run either 60-iteration benchmark batch from the BAE root:

```bash
./.venv/bin/python examples/run_schur_pcg_nesterov_batch.py \
  --solver nesterov --iterations 60 --cache-dir /path/to/bal/files

./.venv/bin/python examples/run_schur_pcg_nesterov_batch.py \
  --solver cg --iterations 60 --cache-dir /path/to/bal/files
```

The compact reference summaries are stored separately in
`python/bae_results/`.
