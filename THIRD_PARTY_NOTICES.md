# Third-party notices

`evals/official_prompt.py` contains the `get_anscheck_prompt` prompt templates from [LongMemEval](https://github.com/xiaowu0162/LongMemEval), commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`, `src/evaluation/evaluate_qa.py`. Python formatting follows this repository's style; all seven prompt outputs were checked against the pinned original source.

Copyright (c) 2024 Di Wu. The original is licensed under MIT; its license text is at `licenses/LongMemEval-MIT.txt`. The copied function is used only in the evaluation module, never in the memory-store package.
