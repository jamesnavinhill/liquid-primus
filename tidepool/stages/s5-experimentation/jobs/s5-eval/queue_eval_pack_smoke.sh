#!/usr/bin/env bash
# The packed evaluation harness earns a job id before s6 depends on it.
#
# Two arms on one L4, both on the published checkpoint, splitting the four components between
# them at sixteen items each. Small on purpose: what is being tested is the packing, not the
# scores. It exercises every mechanism s6 needs -- per-arm config through pack_overrides, two
# torch processes under separate ceilings on one card, per-arm output directories, per-arm
# benchmark caches, and score.json collected back per arm -- and finishes in minutes.
#
# The one thing it does not exercise is the 4-bit path, which needs a built serving object and
# a second port. That is checked when the first Q4 row is run, and the port derivation is
# covered statically by test_pack_isolation.py in the meantime.
set -e
lab task queue 08b4e028-b3b7-45a5-9bed-9f487c9c95ed -e tidepool --no-interactive --provider aws \
  -p arms=E1,E2 \
  -p pack_gb=10,10 \
  -p pack_headroom=0.9 \
  -p stagger_seconds=30 \
  -p run_tag=s5.3-evalpack-smoke \
  -p limit_per_component=16 \
  -p pack_overrides='{"E1": {"components": "bfcl,probes"}, "E2": {"components": "ifstruct,ifeval"}}' \
  -m 'Packed evaluation smoke: two arms of the s5.2 harness on one L4, four components split between them at 16 items each. Proves the harness runs as a packing child -- per-arm config, per-arm output and benchmark caches, two ceilings on one card, score.json per arm -- before s6 scores eight checkpoints this way.'
