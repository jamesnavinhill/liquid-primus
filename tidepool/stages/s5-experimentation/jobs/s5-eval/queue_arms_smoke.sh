#!/usr/bin/env bash
# Four trained arms on one L4, sixteen items a component, before the full pass.
#
# It exists for two reasons, neither of which is a score. First, `C3` is a full-parameter arm:
# its archive holds a `config.json` and a `model.safetensors` and no `adapter_config.json`,
# because tuning every weight leaves nothing to adapt. The harness now recognises that shape
# and loads it as the model rather than merging it onto the base, and that path has never run
# against a real checkpoint. Second, the first eval pack was sized by guessing 10 GB an arm on
# a 24 GB card, which fits two; the harness now reports peak memory, so this run measures what
# an arm actually holds and the full pass is sized on it instead of on another guess.
#
# Sixteen items a component keeps it to a few minutes. The numbers are not comparable with the
# s5.2 baseline rows and are not to be quoted next to them.
set -e
lab task queue 08b4e028-b3b7-45a5-9bed-9f487c9c95ed -e tidepool --no-interactive --provider aws \
  -p arms=C1,C2p,C3,C7 \
  -p pack_gb=5.2,5.2,5.2,5.2 \
  -p pack_headroom=0.9 \
  -p stagger_seconds=30 \
  -p run_tag=s5.3-arms-smoke \
  -p limit_per_component=16 \
  -p pack_overrides='{"C1": {"adapter_object": "tidepool/s5.3/arms/C1/adapter.zip"}, "C2p": {"adapter_object": "tidepool/s5.3/arms/C2p/adapter.zip"}, "C3": {"adapter_object": "tidepool/s5.3/arms/C3/adapter.zip"}, "C7": {"adapter_object": "tidepool/s5.3/arms/C7/adapter.zip"}}' \
  -m 'Four-arm loadability and sizing smoke: C1, C2p, C3, C7 scored on all four components at 16 items each, four arms on one L4 under 5.2 GB ceilings. Proves the full-parameter checkpoint loads as a model rather than as an adapter, and measures peak memory per arm so the full scoring pass is sized on evidence.'
