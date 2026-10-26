# Vendored OC-SORT

- Upstream: https://github.com/noahcao/OC_SORT
- Pinned commit: `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- Downloaded: 2026-10-22
- License: upstream MIT `LICENSE`; the Kalman filter file also retains its
  original embedded copyright and license notice.

The following files are exact downloads from that commit, without edits:

- `trackers/ocsort_tracker/ocsort.py`
- `trackers/ocsort_tracker/association.py`
- `trackers/ocsort_tracker/kalmanfilter.py`
- `LICENSE`

See `PROVENANCE.json` for individual source URLs and SHA256 digests. Package
marker `__init__.py` files and this provenance documentation are local additions.
Application adaptations are confined to `../../ocsort_adapter.py`.

Runtime dependencies for these algorithm files are NumPy, SciPy and FilterPy.
The upstream association function natively uses SciPy's
`linear_sum_assignment` when optional `lap` is unavailable; this project uses
that existing fallback and does not require a LAP compiler on Windows.

OC-SORT is used as a tracker, not as a detector or age classifier. The adapter
uses the official OC-SORT with IoU association, inertia `0.2`, `delta_t=3`, and
`use_byte=False`; it does not substitute another tracking algorithm. Every
frame advances official `update`, including empty detections. The official
`min_hits` return filter is supplemented by independent detection counts and
source-video timestamps for selection. Missed tracklets remain diagnostic
predictions only, and are never new person/pose evidence.
