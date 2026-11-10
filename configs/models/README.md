# Model declarations

One declaration per model or baseline. Record the upstream implementation and
revision, local adapter revision, point/population capability, supported task
and perturbation modalities, environment, external assets, random seeds, and
cluster resource defaults.

Capability is explicit: a point model is not silently converted into a
population model solely to make it eligible for distributional metrics.

The initial ranking-sensitivity study freezes two substantive models: GEARS for
point prediction on unseen genetic perturbations, and STATE for native
population prediction across genetic and chemical tracks. No-change,
training-condition mean, and—where gene embeddings make sense—the bilinear PCA
baseline anchor the rankings. The interpolated technical duplicate remains a
positive control, not a ranked model.

