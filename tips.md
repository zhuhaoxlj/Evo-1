# Evo-1 训练量差异简析

  

作者 README 里的训练配置是 Stage 1 跑 `5000 steps`，Stage 2 跑 `80000 steps`，但作者实际训练时是 `8 卡 A100`，每卡 `batch size=16`，所以全局 batch size 是 `8 × 16 = 128`。你本地单张 4090 如果用 `batch size=16`，全局 batch size 只有 `16`，因此同样 step 下有效训练数据量只有作者的 `16 / 128 = 1/8`。

  

这个结论不是主要从论文 PDF 推出来的，而是由三部分信息综合得到：`README.md` 里的 Stage 1/Stage 2 命令、本地 `run_cp_gpu_train_swanlab.sh` 的默认参数，以及你和作者沟通中提到的“8 卡 A 100、每卡 batch size 16”。换算公式是：`有效样本数 = global_batch_size × max_steps`。

  

按单卡 4090、`batch size=16` 换算，如果想接近作者训练量，Stage 1 应从 `5000 steps` 增加到约 `40000 steps`，Stage 2 应从 `80000 steps` 增加到约 `640000 steps`。不过 Stage 2 的 `640000 steps` 很长，工程上更建议先跑 `160000` 或 `320000 steps` 看 rollout 成功率是否明显提升。

  

所以你之前 loss 曲线好看但 MetaWorld 成功率低，比较可能的原因之一是：单卡训练虽然 step 数看起来一样，但实际看过的数据量远少于作者配置，尤其 Stage 2 full-scale training 没有充分等效。