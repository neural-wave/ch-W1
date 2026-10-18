# Use of AI coding assistants

No pretrained model, external weights or embeddings are part of the solution. The model is
PCHIP interpolation plus three LightGBM regressors trained from scratch on the official data,
followed by fixed mean-preserving post-processing rules.

AI assistants were used as development tools only:

| Tool | Used for | Human review |
|---|---|---|
| Claude Code | Code drafting and refactoring, tests, documentation, repository restructuring, dashboard | Changes kept only after passing the test suite and the development backtest; documentation and report checked against the results files |
| OpenAI Codex | Feature-engineering review notes, first drafts of the report | Suggestions tested on the development folds before adoption; report text rewritten and checked by the team |
| Agent-driven experiment loop | Automated runs of the fixed backtest harness on the official training data to compare candidate configurations (this is how the I3 post-processing rules were found) | Candidates promoted only through the predefined development gate; the switch from I2 to I3 was decided by the team ([METHODOLOGY.md](METHODOLOGY.md) section 4) |

We used the assistants iteratively: we tried an approach, looked at the results, and used them to
help find errors and work through possible solutions. We also discussed EDA findings with them to
decide which features or model changes were worth testing, then checked those suggestions through
tests and backtesting, keeping only the changes that improved the results.

All experiment results reported in [results/summary.csv](../results/summary.csv) come from the
code in this repository and can be regenerated with the commands in
[results/README.md](../results/README.md). See also [PROVENANCE.md](PROVENANCE.md).
