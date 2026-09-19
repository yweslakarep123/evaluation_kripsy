Kettle action PCA — FlowPolicy NFE hypothesis
=============================================

Figure: 01_kettle_action_pca.{png,pdf}
Table:  kettle_action_pca_distances.csv

Method
------
- Demo actions: kettle windows from data/kitchen/*_seq.npy,
  segmented by object-goal completions (BONUS_THRESH=0.3).
- Policy actions: executed_action in kettle windows from
  kitchen_eval_nfe100 FlowPolicy trajectory_logs (seeds 42/43/44),
  only episodes that completed kettle; window from task_durations_ms.
- PCA(2) fit on demo kettle actions only, then transform all groups.
- Ellipses: solid = 1σ, dotted = 2σ (per NFE); dashed gray = demo 2σ.

PCA variance explained: PC1=65.9%, PC2=16.9%,
  cumulative=82.8%

Mean Euclidean distance to demo centroid (action space, 9-D)
-----------------------------------------------------------
  NFE=1    n=12273    mean_dist=4.226176
  NFE=8    n=5226     mean_dist=4.096034
  NFE=32   n=5688     mean_dist=4.069586
  NFE=100  n=5169     mean_dist=4.078541

NFE1→NFE8 distance trend: NOT increasing
  (increasing supports drift-from-demo hypothesis;
   decreasing/stable weakens it)

Limitation: only kettle-SUCCESS episodes contribute policy points;
path-selection failures (never attempting kettle) are excluded by design.
