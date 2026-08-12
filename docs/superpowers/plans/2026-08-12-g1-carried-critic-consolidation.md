# G1 Carried-Critic Consolidation Plan

1. Load and validate the exact E021 30-phase dataset and parent artifacts.
2. Normalize with E012's frozen critic normalizer and execute exactly 1,640
   full-dataset continued-Adam critic steps at `5e-4`.
3. Capture only the untouched confirmation phases under the frozen actor and
   compare original target versus consolidated critic on the same returns.
4. Preserve every non-critic TrainState field, publish checkpoint only on all
   calibration/improvement gates, and write the report last.
5. Run focused tests/static checks, independent review, registry validation,
   and one guarded GPU execution before curating the outcome.
