# Changes vs upstream (kevinghst/HWM_PLDM main)

Full diff: `changes.diff` (generated via `git diff upstream/main lang-align`).
This branch adds one thing: a contrastive loss (`LangAlign`) that aligns the
level2 latent action `l_t` with sentence embeddings of rule-based captions
describing each waypoint segment's movement. See
`pldm/objectives/lang/lang_align.py` for the motivation in full.

## New files

- **`pldm/objectives/lang/captioner.py`** — Maps waypoint-anchor coordinate
  deltas (already present in the data pipeline as `batch.l2_locations`) to a
  small fixed vocabulary of direction+magnitude captions (e.g. "move a long
  distance to the northeast"), as integer ids. No data pipeline changes
  needed; pure tensor math, no runtime string generation.

- **`pldm/objectives/lang/text_encoder.py`** — `CaptionEmbedder` turns the
  fixed caption vocabulary into a `(K, embed_dim)` embedding table, either
  frozen (encoded once and cached) or jointly fine-tuned at a low learning
  rate. Uses `transformers` `AutoModel`/`AutoTokenizer` + manual mean pooling
  instead of `sentence-transformers`' `.encode()`, which can silently run
  under `no_grad` in some versions and would break the fine-tune path.

- **`pldm/objectives/lang/lang_align.py`** — `LangAlignObjective`, the loss
  itself. Because the caption vocabulary is small, a batch has many segments
  sharing a caption, so plain CLIP-style InfoNCE would treat same-caption
  pairs as false negatives; this uses exact K-way cross-entropy for the
  z→text direction and SupCon (multi-positive) for text→z, plus a dispersion
  penalty on the text prototypes to guard against encoder collapse in
  fine-tune mode. Follows the existing `objectives/` conventions exactly
  (same `__call__(batch, results)` signature, `NamedTuple` LossInfo with
  `build_log_dict()`, explicit `.cuda()` since objectives aren't submodules
  of the trained model).

- **`pldm/objectives/lang/__init__.py`** — Just a module docstring pointing
  back at this branch/fork for provenance; no code.

- **`pldm/configs/diverse_maze/lang/lang_align_overlay.yaml`** — Thin config
  layer meant to be passed alongside the untouched baseline yaml via
  `--configs base.yaml lang_align_overlay.yaml`, so the original baseline
  config never needs editing. Adds `LangAlign` to `objectives_l2.objectives`
  and sets its default hyperparameters.

## Modified files

- **`pldm/objectives/__init__.py`** — Registers `LangAlign` the same way
  every other objective type is registered: an `ObjectiveType` enum member,
  an `ObjectivesConfig.lang_align` field, and a `build_objectives_list()`
  branch. That function also gained two new optional kwargs (`location_std`,
  `z_dim`) that only the `LangAlign` branch consumes — every other branch is
  untouched.

- **`pldm/train.py`** — Passes `location_std`/`z_dim` into the `objectives_l2`
  build call (the `objectives_l1` call is untouched; `LangAlign` is
  level2-only), and collects `objective.param_groups()` from every l1/l2
  objective to forward into `OptimizerFactory`.

- **`pldm/optimizers/optimizer_factory.py`** — Accepts an
  `extra_param_groups` list, appended to both the LARS and Adam code paths.
  This matters beyond `LangAlign`: objectives that own learnable modules
  (e.g. `VICRegObjective`'s `Projector`) were never actually registered with
  the optimizer before this change, since it only walks
  `model.level1`/`model.level2`'s parameters and objectives are standalone
  objects, not submodules of the trained model. `LangAlignObjective`'s and
  `CaptionEmbedder`'s `param_groups()` also set `"base_lr"` (not just
  `"lr"`) on their groups, because `pldm.optimizers.schedulers.Scheduler`
  recomputes `"lr"` from `"base_lr"` every step, defaulting to the
  model-wide `base_lr` for any group missing it — without this, the
  fine-tuned text encoder's intended low LR would be silently overwritten on
  the very first scheduler step.

- **`requirements.txt`** — Adds `transformers` (used by `CaptionEmbedder`;
  not previously a dependency of this repo).
