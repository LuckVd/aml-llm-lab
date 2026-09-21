# Lab 04: Scaling Law 拟合（预置实验数据）

> 对应 CS336 Adapter（Scaling Laws） / 唐杰课程作业④
> 预计耗时：半天　|　资源：纯 CPU，秒级　|　依赖：numpy + torch（无训练）
> **数据来自 `presets/`（合成数据，见 presets/README.md 诚实标注）——本 lab 拟合与外推，不跑训练**

## 你要做什么

在 `scaling.py` 里实现四个函数：

1. `flops(n_params, n_tokens)`：训练计算量 6ND
2. `chinchilla_loss(n, d, params)`：损失面 L(N,D) = E + A/N^α + B/D^β
3. `fit_scaling_law(runs)`：用梯度下降把 (A, α, B, β) 从 56 个预置数据点里拟合回来（E 给定 1.7）
4. `compute_optimal(budget_flops, params)`：给定算力预算，返回最优 (N*, D*) 及 D*/N*

## 文件

| 文件 | 说明 |
|---|---|
| `scaling.py` | **你要写的文件**，所有 `TODO` 都在这里 |
| `tests/test_scaling.py` | 测试：拟合参数要能恢复生成真值（±10% 内） |
| `solution/scaling_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_scaling.py` | 拟合曲线、计算最优前沿、定位 lab03 的 20M 点 |
| `../../book/04-scaling.md` | 原理讲解（先读这个） |
| `../../presets/` | 预置数据 + 生成脚本 |

## 通过标准

```bash
make lab4
```

1. pytest 全绿（含：从合成数据恢复 Chinchilla 量级参数、D*/N* 落在 10–40、MFU 计算）
2. demo 里能看到：拟合参数 vs 文献值对照表、各预算档位的最优 N/D、lab03 那个"严重数据不足"的点

## 思考题（demo 末尾会用到）

- 20M 模型在 1.6M token 的点上，离同预算的计算最优 loss 差多少 nat？
- 把预算放大 100 倍，最优参数量放大几倍？（>10 倍还是 <10 倍？幂指数是多少？）
- MFU=0.6 是 A100 上的好成绩。你 lab03 的 CPU "MFU" 是多少（peak 按 100 GFLOPS 估）？
