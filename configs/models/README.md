# Model declarations

One declaration per model or baseline. Record the upstream implementation and
revision, local adapter revision, point/population capability, supported task
and perturbation modalities, environment, external assets, random seeds, and
cluster resource defaults.

Capability is explicit: a point model is not silently converted into a
population model solely to make it eligible for distributional metrics.

The perturbation-generalization panel freezes six substantive models: GEARS,
PRESAGE, scGPT, scLAMBDA, STATE, and TabICLv2. No-change, training-condition
mean, and the bilinear PCA baseline anchor the rankings. The interpolated
technical duplicate remains an oracle control, not a ranked model.

The context-generalization panel is not yet frozen. STATE is confirmed; trVAE,
scGen, scPRAM, and scVIDR are under feasibility review using scArchon's published
implementations. See `docs/study-design.md` for the decision record and the
adapter-scaling constraint.

Hugo's six existing model environments remain in Pixi. PerturbationData and the
evaluation/orchestration layer use `uv`; model declarations must record the
environment or container revision used for each run.

