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

The context-generalization panel uses STATE, CPA, bioLORD, scDisInFact, and
trVAE for both genetic and chemical context transfer. trVAE remains conditional
on a successful jointly perturbation-conditioned smoke test; failure removes it
rather than changing the task to one fit per perturbation. No-change,
training-condition mean, and a context-aware linear model are ranked baselines.

Hugo's six existing model environments remain in Pixi. New model runners also
receive one Pixi environment per model, reconstructed from pinned reference
containers where appropriate and checked with a small equivalence test.
PerturbationData and the evaluation/orchestration layer use `uv`; declarations
must record both the Pixi lock and pinned reference revision or image digest.

