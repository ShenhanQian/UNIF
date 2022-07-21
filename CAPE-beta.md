# CAPE-beta

The CAPE dataset does not provide the shape parameters (beta) for each subject. We fit the parameters on our own with the following command:

```shell
python main.py \
--cfg config/cape_subjects_bodyshape.py \
EXP.tag 00032 \
DATASET.kwargs.subject_name 00032 \
```

This program optimizes the beta parameters of SMPL to reconstruct the vertices in the file: `data/cape_release/minimal_body_shape/<subject_name>/<subject_name>_minimal.npy`.
