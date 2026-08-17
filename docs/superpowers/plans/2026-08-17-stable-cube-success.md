# Stable Cube Success Implementation Plan

1. Add ORCA Sim regression tests for height, velocity, ten-step hold, and drop priority.
2. Implement cached per-step task status and expose stability diagnostics in `info`.
3. Thread the thresholds through ORCA Train configuration and CLI.
4. Run both test suites, sync both repositories to Spark, smoke-test, then launch a fresh run.
