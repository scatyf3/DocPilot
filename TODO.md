# TODO

## Evals

现状（2026-11-01）：检索已在公开数据集上测过，但回答层（correctness、citation faithfulness、refusal）还没有真实结果。README 的回答指标还是 TBD，`evals/baseline.json` 由 fake embedder 生成，CI 没有配置密钥，走的是 fake 回退。

模型用默认配置即可：embedding 用 `text-embedding-3-small`，回答模型和 judge 都用 `claude-opus-5-5`。要测的是实际部署的配置，不是最强的模型。

### P0：跑出第一批真实数字

- [ ] 在 `.env` 配好 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY` 和 `DATABASE_URL`。
- [ ] 试跑，确认流程没报错，并记下成本：`python -m evals.run --embedder openai --llm anthropic --limit 10`
- [ ] 全量跑一次（去掉 `--limit`）。
- [ ] 人工抽查 judge：`python -m evals.spotcheck evals/results/<run>.json -n 20`。如果和人工判断分歧明显，先修 judge，再用这批数字。
- [ ] 按 single、multi、unanswerable 分类看错题，定位是检索、回答还是 judge 的问题。
- [ ] 填上 README 的 TBD。

### P1：让数字可信

- [ ] 全量重复跑 3 次，看波动有多大。之后小于这个波动的差异都不算变化。
- [ ] 在结果文件里记录 commit SHA。
- [ ] 用真实配置重新生成 `evals/baseline.json`，同时配好 CI secrets。两件事要一起做，否则 CI 会拿 fake 结果和真实基线比。
- [ ] 控制 CI 成本：每个 PR 只跑检索，回答评测改成打 label 或手动触发。
- [ ] 加回归门禁，让 `evals/compare.py` 的结果能让 CI 失败。阈值根据重复运行的波动来定。

### P2：让结果更有说服力

- [ ] 把拒答题从 11 道加到 30 道以上。现在错一道，拒答准确率就差约 9%。
- [ ] 加一套更难的回答评测，比如从 MultiHop-RAG 抽 100 题。现有 wiki 只有 32 个 chunk，分数可能普遍偏高。
- [ ] 对比 Haiku、Sonnet、Opus 的质量、延迟和成本。
