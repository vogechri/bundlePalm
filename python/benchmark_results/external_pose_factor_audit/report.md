# Transient External Pose-Factor Audit

Date: 2026-08-13

## Intended Contract

Add one scene-independent relative rotation/translation continuation during
basin acquisition, anneal it completely away, and evaluate the unchanged pixel
objective on Roman Forum, Trafalgar, and BAL1490.

## Available Information

The 1DSfM conversion retains genuine pairwise initialization measurements:

- Roman Forum `translation_graph.npz`: 94,250 edges;
- Trafalgar `translation_graph.npz`: 861,509 edges;
- each graph stores original camera-ID pairs, translation directions, weights,
  and a retained mask;
- `cleaned_selection.npz` stores the original IDs retained as compact BAL
  cameras, permitting an exact original-ID to BAL-index map;
- relative rotations remain recoverable from the epipolar-geometry file and
  `global_rotations.npz`.

Standard BAL input has no corresponding information. Its schema contains only
camera/point counts, indexed pixel observations, initial 9-parameter cameras,
and initial 3D points. No independent pairwise pose measurements or BAL1490
pose sidecar exists in this workspace.

## Existing Experiment

The proposed 1DSfM mechanism already exists in
`serverTest/coarse_graph_schur_refine.py`. It loads retained translation
directions and relative rotations, applies a robust coupled prior in eight low
camera-graph modes with frozen weight `0.1`, and removes it before ordinary DRS.
All final states are evaluated on the unchanged pixel objective.

The completed all-15 fixed-policy gate is unsafe:

- geometric SSE ratio: `0.989513x`;
- summed SSE ratio: `1.001376x`;
- W/T/L: 9/0/6;
- worst regression: Trafalgar `1.050620x`.

A loss-free physical-SSE branch decision first appears after Schur correction
9, at approximately `1.900388x` balanced-workflow time. Earlier development
selectors reverse on held-out Notre Dame and Yorkminster. This closes fixed
continuation and cheap pixel-SSE selection for the available external graph.

## Rejected Substitutes

1. Deriving relative factors from each BAL file's initial cameras is globally
   available but is not external pose evidence. It anchors the optimizer to its
   own initial state and overlaps existing proximal/trust continuation.
2. Reconstructing pairwise poses from BAL pixel observations uses the same
   measurements as the evaluated objective, adds a new two-view estimator, and
   still does not match the retained 1DSfM initialization contract.
3. Enabling measured factors only on 1DSfM violates the one-policy,
   cross-family gate because BAL1490 would run a different algorithm.

## Decision

Do not add transient pose factors to production DRS under the current data
contract. The valid 1DSfM version is already retained as a default-off basin
proposal; BAL lacks the independent information required for the requested
cross-family continuation. Reopen only when every benchmark family supplies a
persisted, camera-ID-mapped relative-pose graph with common semantics.
