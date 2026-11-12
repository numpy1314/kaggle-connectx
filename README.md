# Kaggle ConnectX

`submission.py` 是可直接上传的智能体，面向 Kaggle 默认的 6 行、7 列、4 连棋盘。文件内嵌 Linux x86-64 原生搜索引擎、开局库和纯 Python 回退。智能体每回合接收棋盘与执子方，返回从 0 开始编号的列。

## 提交

在 ConnectX 比赛页加入比赛并接受规则，上传 `submission.py`。Kaggle 先运行自对弈验证，随后安排与其他提交的对局。排行榜分数随平台对局更新，每天最多提交两次。[比赛说明](https://www.kaggle.com/competitions/connectx) · [比赛规则](https://www.kaggle.com/competitions/connectx/rules)

已配置 Kaggle CLI 的用户也可以在本目录运行以下命令。[官方 CLI 文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md)

```bash
kaggle competitions submit connectx -f submission.py -m "Native search and proven openings"
```

## 策略

- **开局**。先手首着占据第 4 列。该策略覆盖至棋盘已有 16 枚棋子时的全部可达先手局面，保有制胜走法直至先手第 9 次落子。后手遇到对方中列开局时选第 3 列，这一回应来自本地对照对局。
- **中残局**。先处理一步获胜、必须封堵和会送给对手一步胜利的走法。用位棋盘枚举可行着法，用精确搜索证明胜负或保和。在需要截断的局面，以迭代加深的 Negamax 搜索、α–β 剪枝和置换表选着。
- **用时**。常规回合的内部时间预算上限为 1.65 秒，按环境提供的 `actTimeout` 缩减。搜索到达截止时间时返回上一轮完整搜索的合法走法。其他尺寸的棋盘或原生库加载失败时使用纯 Python 搜索。[Kaggle 智能体时限](https://www.kaggle.com/c/connectx/overview/environment-rules)

`openings.json` 记录 65,513 个左右镜像去重的局面。`history` 是从 0 开始编号的落子序列，`move` 是选中的列。`outcome` 为 `1` 表示该走法可证胜，为 `0` 表示可证保和，为 `null` 表示采用本地对照结果选择的后手开局回应。开局库由本项目的搜索引擎离线生成。

## 验证结果

- Kaggle 官方测试包自对弈正常结束。对内置 Random 和 Negamax 各换边对弈 12 局，两组均全胜。两组共测量 214 次落子，单步最长约 1.651 秒。
- 对去掉开局库的同一引擎，在每步 0.35 秒的设置下换边对弈 24 局，成绩为 15 胜、3 和、6 负。
- Linux x86-64 测试验证了提交文件能加载内嵌原生引擎和开局库，并完整下完换边对局。

## 复现

`agent.py` 和 `engine.c` 分别是智能体入口及原生引擎的源码。`build_submission.py` 将引擎和开局库打包成单文件。构建可使用 Linux x86-64 上的 C 编译器，或 macOS 上的 `clang` 和 `ld.lld`。

```bash
python3 build_submission.py
python3 benchmark.py verify
python3 benchmark.py play --opponent no-book --pairs 12 --timeout 0.35
```

`verify` 用独立棋盘逻辑核对合法走法和胜利判定，用穷举搜索核对 600 个残局，并遍历先手开局覆盖。`play` 从相同的前缀各交换一次先后手，与去掉开局库的同一引擎比较。

通过已安装的 `kaggle-environments` 复现自对弈和官方基线对局。项目验证使用了该包的 `1.9.11` 版本。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install kaggle-environments==1.9.11
.venv/bin/python benchmark.py kaggle --pairs 6
```

如需继续预计算，下面的命令保留现有开局并检查先手策略中的遗漏。`--scale` 调整每个待求解局面的预算。

```bash
python3 build_openings.py --seed openings.json --plies 10 --first-win-plies 16
python3 build_submission.py
```
