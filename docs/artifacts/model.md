# EgoStitch 模型：沿流程图理解每个模块为什么存在

配套流程图见 [egostitch-intro.html](./egostitch-intro.html)。

这份讲解不展开逐层参数、完整 loss 公式或工程实现细节，而是沿流程图从上到下回答两个问题：

1. 每个模块大致在做什么？
2. 为什么模型需要这样设计？

---

## 0. 一句话理解整个模型

> EgoStitch 最终不是要生成一张图，而是要判断 queried pair $(u,v)$ 之间是否存在边。它先为两个 endpoint 各自构造一个“可能的局部邻域摘要”，再把两个局部邻域拼起来，用生成的 topology 帮助最终二分类。

整条路径可以压缩成：

> 读取两个节点本身
>
> → 想象它们各自可能处在怎样的局部邻域
>
> → 比较并拼接两个邻域
>
> → 判断两者是否应该连接

这里的 generated topology 只是中间上下文，不是模型的最终输出。

---

## 1. 三类输入：同一个节点的三种观察方式

对于这个 retrieval-grounded EgoStitch 实验臂，每个 endpoint 使用三类输入。

### 1.1 Raw token embeddings

这是节点最细粒度的原始表示。模型保留完整 token sequence，而不是一开始就把所有信息压成一个向量。

**它做什么？**

让模型看到节点内部不同 token 所表达的细节。

**为什么这样设计？**

两个节点是否连接，可能只取决于少数关键 token。过早做平均会丢失这些局部信号，因此 raw-token branch 一直保留到后面的 pair cross-attention 和 readout。

### 1.2 Mean-pooled node embedding

这是把一个节点的所有 token 平均后得到的整体摘要，也就是流程图中的 $x_u,x_v$。

**它做什么？**

提供一个固定长度的“节点名片”，供 grounding retrieval 和 neighborhood generator 使用。

**为什么这样设计？**

Generator 需要稳定、固定宽度的条件输入，不适合直接处理长度可变的完整 token sequence。Mean pooling 在这里承担的是检索和生成条件，而不是最终 pair reasoning。

### 1.3 Grounding pool

模型为每个 endpoint 从同侧 node universe 中检索 top-50 相似节点，只读取这些节点的 features。

**它做什么？**

为 generator 提供一组现实中的参考候选：与当前 endpoint 在 feature space 中相似的节点通常是什么样的？

**为什么这样设计？**

如果完全让 generator 从一个向量中凭空创造所有 neighbors，生成空间会很大，也很难稳定训练。Grounding pool 给它提供现实锚点，同时仍允许模型偏离这些候选，生成更抽象的 imagined slots。

Grounding pool 不包含候选节点的边、标签或真实 ego graph，因此它提供的是 feature reference，不是 target-topology leakage。

---

## 2. Siamese endpoint encoder：先理解两个节点各自是什么

Raw token embeddings 首先分别进入同一个 endpoint encoder。两个方向共享参数，所以称为 Siamese encoder。

**它做什么？**

在每个 endpoint 内部，让 tokens 互相交流，形成带上下文的 token representations。

**为什么这样设计？**

在讨论 $u$ 和 $v$ 的关系之前，模型需要先理解每个节点自身。共享 encoder 还保证两个 endpoint 被映射到同一种表示空间，避免模型因为它们位于输入左侧或右侧而采用不同的理解规则。

这一阶段两个 endpoint 还没有直接交互。

---

## 3. Feature standardization：给 generator 一个稳定的坐标系

Mean-pooled endpoint feature 和 grounding features 在进入 generator 前使用训练侧统计量做标准化。

**它做什么？**

让不同 feature dimensions 处在更可比较的尺度上。

**为什么这样设计？**

如果某些维度天然波动很大，它们会不成比例地影响 retrieval、projection 和 slot generation。冻结训练侧统计量还能保证 validation、test 和 deployment 使用同一坐标系，而不是根据测试数据重新定义 feature geometry。

---

## 4. Shared endpoint generator：把节点变成一组 imagined neighbor slots

Generator 对 $u$ 和 $v$ 分别运行，但两侧共享参数。每个 endpoint 最终产生 16 个 slots。

可以把一个 slot 理解成：

> 一类可能存在于该 endpoint 周围的 neighbor group。

它不是一个确定的真实 node ID，也不一定只代表一个邻居。

### 4.1 Endpoint summary

Generator 先把 endpoint feature 压缩成一个紧凑的 endpoint code，并估计这个节点大概需要多大的 neighborhood budget。

**为什么需要它？**

不同节点的局部结构复杂度不同。有些节点可能只有少量邻居，有些节点则需要更多 neighbor mass。Generator 不能假设所有节点拥有同样大小的 ego network。

### 4.2 Grounding-aware slot queries

16 个 slot queries 从不同 grounding candidates 出发，同时读取 endpoint summary 和整个 grounding pool。

**它做什么？**

让不同 slots 以不同现实候选为起点，再根据当前 endpoint 共同调整。

**为什么这样设计？**

如果所有 slots 都从同一个 endpoint vector 出发，它们很容易生成相似内容并发生 slot collapse。不同 grounding seeds 提供初始分工，而 slots 之间的交互使它们能够协调、去重和重新组织。

### 4.3 每个 slot 输出什么？

每个 slot 不只产生一个 content vector，还描述：

- 这个 slot 是否应该存在；
- 它大致代表多少相似邻居；
- 它是否应由真实 grounding candidate 支撑；
- 如果需要 grounding，它更像哪一个 candidate；
- 它与同侧其他 slots 之间可能如何连接。

**直觉**

Generator 生成的不是 16 个互相独立的假节点，而是一个带容量、可信度和内部连接的 compressed ego-network sketch。

---

## 5. Generator 周围的三类训练压力

流程图中的 loss 1、2、3 都在回答同一个问题：

> Generator 产生的 imagined neighborhood 是否既像真实邻域，又足够稳定？

### Loss 1：reconstruction

训练时，模型把 generated slots 与真实训练邻域中的 neighbors 做 permutation-invariant matching。

**它教什么？**

- slot content 应该接近真实 neighbor 类型；
- 需要的 slots 应存在，不需要的 slots 应关闭；
- multiplicity 和 degree budget 应合理；
- 同侧 slot adjacency 应反映真实局部关系；
- grounded slot 应选择正确的候选；
- 不同 slots 不应全部坍缩成同一种表示。

**为什么不是固定 slot-to-neighbor 对应？**

Slot 本身没有天然顺序。第一个 slot 不一定永远代表同一种 neighbor，因此训练前必须先寻找最合理的匹配。

### Loss 2：realism

Reconstruction 关注单个 slots 是否匹配，但单点正确不保证整张 ego sketch 合理。Realism loss 从整体上比较 generated 和 real ego networks。

**它教什么？**

让生成邻域在 degree、density、内部连接模式以及图级表示上更接近真实邻域分布。

**为什么需要它？**

一个模型可能逐 slot 看起来都不离谱，但组合起来形成不真实的网络。Realism loss 负责约束这种整体结构。

### Loss 3：self-supervised consistency

模型轻微扰动 endpoint feature 或 grounding pool，然后要求 ungrounded imagined slots 不要发生剧烈变化。

**它教什么？**

让真正依赖模型想象的部分对小扰动保持稳定。

**为什么主要约束 ungrounded slots？**

Grounded slots 已经有现实候选和 reconstruction target 作为锚点；完全 imagined 的 slots 更容易漂移，因此更需要 consistency regularization。

---

## 6. Pair cross-attention：让两个真实 endpoint 直接交流

在 generator 之外，raw-token branch 让 $u$ 和 $v$ 的 token representations 进行多轮 cross-attention。

**它做什么？**

让模型根据 counterpart 重新判断本侧哪些 token 最重要，并形成一个 pair-level CLS representation。

**为什么不能只依赖 generated topology？**

Generated neighborhood 是压缩且不确定的中间表征。两个 endpoint 自身的直接内容匹配仍然是最基础的证据，不能被想象的 topology 取代。

因此 EgoStitch 始终保留一条强的 pairwise content backbone；topology 是在此基础上的补充。

---

## 7. Generator 输出后为什么只保留 topology conditioning？

Generator 的 slot set 只通过 typed scaffold 进入最终 classifier：它回答
“这些 imagined neighbor groups 如何连接”。完整 slot content 仍用于 generator
自身的重建与训练损失，但不再作为独立 conditioning pathway 输入 edge head。
这样 classifier 无法绕过结构、直接凭 slot content 完成预测。

---

## 8. Align the two slot sets：寻找两个 imagined neighborhoods 的对应关系

$u$ 的第一个 slot 与 $v$ 的第一个 slot 没有天然对应关系，因此模型不能直接按 slot index 比较两侧。

Sinkhorn alignment 会根据 slot content 和 represented neighbor mass，形成一个软对应矩阵。

**它做什么？**

回答：

> $u$ 侧的这个 neighbor group，最可能对应 $v$ 侧的哪些 neighbor groups？

一个 slot 可以与多个 slots 部分对应，也可以没有完美匹配。

**为什么用软对齐？**

两侧 imagined neighborhoods 的粒度可能不同。一个大 slot 可能对应另一侧多个小 slots，强行做一对一 hard matching 会过早丢失不确定性。

### Loss 4：alignment supervision

训练图中如果两侧真实邻域共享同一个 neighbor identity，模型就知道哪些 generated slot pairs 应承载更多 alignment mass。

**它教什么？**

让软对齐不只是“feature 看起来相似”，而是逐渐学会表达真正的 shared-neighbor correspondence。

---

## 9. Stitch typed scaffold：把两个局部世界拼成一张联合软图

有了两侧 slots 和 alignment，模型构建一张包含 34 个节点的 scaffold：

- 两个 endpoint centers；
- $u$ 的 16 个 slots；
- $v$ 的 16 个 slots。

Scaffold 使用四类关系：

| 关系 | 直觉 |
|---|---|
| star | slot 属于哪一个 endpoint |
| intra | 同侧 imagined neighbors 如何互联 |
| align | 两侧哪些 slots 可能对应 |
| closure | 两侧内部连接与跨侧对应是否相互支持 |

**它做什么？**

把“两个独立的局部邻域”变成“一个围绕 queried pair 的联合局部世界”。

**为什么需要 typed edges？**

“属于同一 endpoint”“同侧相邻”和“跨侧对应”不是同一种关系。若把它们全部混成普通 adjacency，模型就无法判断一条 message 的结构含义。

Scaffold 是 soft graph：节点存在度、neighbor mass 和边强度都可以是连续值，因此整个路径能够端到端训练。

---

## 10. STEncoder：让 scaffold 中的结构信息传播

STEncoder 是一个 typed message-passing network。

**它做什么？**

让 scaffold 中每个节点分别从 star、intra、align 和 closure neighbors 接收不同类型的消息。经过多轮传播后，每个节点不再只知道自己的局部属性，也知道自己在整个 stitched neighborhood 中扮演什么结构角色。

例如，一个 $u$ 侧 slot 可以逐渐知道：

- 自己是否重要；
- 同侧还有哪些相关 slots；
- 在 $v$ 侧是否存在对应 group；
- 这种对应是否形成 shared-neighbor 或 closure-like pattern。

**为什么输出一组 topology tokens，而不是立刻平均成一个向量？**

不同 queried pairs 可能依赖不同结构位置。有的 pair 关注某个强 shared-neighbor slot，有的关注整体 density。保留全部 topology tokens，可以让下游 pair representation 自己选择应该读取哪些位置。

**最核心的直觉**

> Scaffold 把 topology 画出来；STEncoder 让信息沿这张图流动，从而把“如何连接”变成可被神经网络读取的表示。

---

## 11. Slot content 的边界：训练 generator，不直接条件化 classifier

Slot content、grounding 与匹配信息仍监督 imagined slots，并参与构建 scaffold
的存在概率、multiplicity 与 typed edges；但完整语义向量不会被组装成第二套
classifier tokens。最终 edge prediction 只接收 structure-only scaffold encoding。

---

## 12. Topology update：先用结构修正 pair representation

Raw-token pair branch 已经形成一个初始 CLS。Topology update 让这个 CLS 查询 34 个 topology tokens。

**它做什么？**

根据 generated scaffold 对原始 pair representation 做一个 residual correction。

可以把它理解为：

> 仅看两个 endpoint 的内容，我原本倾向于某种判断；考虑它们可能处在怎样的联合局部结构后，我是否应该调整这个判断？

**为什么使用一个从关闭状态开始学习的 gate？**

训练初期 generated topology 还不可靠。如果一开始就强行注入，噪声可能破坏已经可用的 pairwise backbone。Gate 让模型只有在 topology 逐渐学出有效信号后，才增加它的影响。

---

## 13. 为什么没有独立 content update？

旧设计曾让 CLS 再查询 slot-content tokens；当前三组件模型已删除这条路径。
因此 `pair_topology` 与 `full` 不再是不同模型，前者也从 active arms 中退役。

---

## 14. PairContextGatedReadout：汇总所有与 queried pair 有关的证据

到这里模型拥有：

- topology-conditioned CLS；
- $u$ 的完整 token representations；
- $v$ 的完整 token representations。

Readout 从 raw tokens 中读取三种视角：

- 平均模式：整体上像什么；
- 强激活模式：是否存在少数特别突出的 token；
- counterpart-conditioned 模式：面对当前另一个 endpoint，哪些 tokens 最相关。

**它做什么？**

把高度压缩的 CLS 与仍然保留细节的 endpoint tokens 合并成最终 pair representation。

**为什么需要多个 pooling 视角？**

有些边由整体相似性决定，有些边可能只依赖一个局部 motif。只使用 mean pooling 或只使用 CLS 都可能遗漏另一类证据。

---

## 15. AB/BA symmetric max：保证交换 endpoint 不改变答案

内部 cross-attention 和 scaffold role labels 都有方向，所以模型分别计算：

- $u$ 作为 source、$v$ 作为 destination；
- $v$ 作为 source、$u$ 作为 destination。

然后将两个方向对称合并。

**为什么这样设计？**

目标是无向 edge prediction。模型可以在内部使用方向帮助组织计算，但最终概率不应因为输入顺序变化。

---

## 16. 底部两个 heads：一个帮助训练，一个负责真正评分

### 16.1 Train-only relational head

这个辅助 head 从 topology tokens 预测 common-neighbor 和 Jaccard overlap。

**它做什么？**

迫使 STEncoder 学出的 topology representation 确实包含 neighborhood overlap，而不是只产生一组难以解释的隐藏特征。

**为什么是 train-only？**

它的目标只用于塑造 representation。Inference 时不会读取真实 common-neighbor 或 Jaccard，也不会把这个 head 的输出送进最终 score。

### Loss 5：relational loss

它直接监督上述两个 topology quantities。

**直觉**

Final edge loss 可能允许模型主要依赖 raw content；relational loss 则明确要求 topology branch 学会表达结构关系。

### 16.2 Binary edge head

这是最终真正评分的 head。

它把对称 pair representation 转换成一个 edge probability：

$$
p_{uv}=P(\operatorname{edge}(u,v)=1).
$$

### Loss 6：binary edge loss

它直接监督最终预测是否正确，并对稀少的 positive edges 给予更高权重。

**直觉**

其他 losses 都在修建和校准中间的 imagined topology；binary edge loss 才定义模型最终要完成的科学任务。

---

## 17. 六个 losses 各自解决什么问题？

| Loss | 它主要防止什么问题？ |
|---|---|
| Reconstruction | slots 与真实局部邻域没有对应语义 |
| Realism | 单个 slots 尚可，但组合出的 ego network 不真实 |
| SSL consistency | imagined slots 对轻微输入变化随意漂移 |
| Alignment | 两侧 slots 的软对应没有 shared-neighbor 含义 |
| Relational | topology encoder 被最终模型忽略，没学到 overlap |
| Binary edge | 整个系统没有直接服务最终 edge classification |

可以把训练过程理解为：

1. 先让 generator、slots 和 alignment 学会基本语义；
2. 再加入最终 edge classification；
3. 最后联合调整，使 generated topology 从“像真实邻域”进一步变成“对 edge prediction 有用”。

这也是为什么模型不是只优化一个 BCE：如果只给最终标签，generator 存在许多投机路径，可以产生无法解释或坍缩的中间 topology；辅助 losses 为每个关键模块提供更直接的学习信号。

---

## 18. Training 与 inference 的边界

### Inference 需要

- queried endpoints 的 raw token features；
- 它们的 mean-pooled features；
- deployment-side node universe 的 features，用于构建 grounding pools；
- 训练完成的模型参数和冻结统计量。

### Inference 不需要

- queried endpoints 的真实 ego graphs；
- 真实 neighbors；
- deployment target edges 或训练时的 topology graph；
- common-neighbor/Jaccard targets；
- reconstruction、realism、alignment 等训练 targets。

因此这里的 inductive claim 是：

> 这个实验臂不读取 target edges/labels，但需要当前 role 的 `V_fit`、`V_hold` 或 test feature universe；这是额外 inference support，不是严格任务输入或确定方法。

Generator 也不是在完全没有参考节点的世界中凭空创造 topology。它是在 feature-only grounding 的帮助下，生成对 queried-edge classification 有用的 latent local context。

---

## 19. 一段可直接用于 oral presentation 的总结

> 对于一对未见节点，EgoStitch 同时保留 endpoint raw-token cross-attention 和 generated local context。模型先从同侧 feature universe 检索相似节点作为 grounding，再为每个 endpoint 生成 16 个 compressed neighbor slots。两侧 slots 经过软对齐，被拼成一张包含 star、intra、align 和 closure 关系的联合 scaffold。STEncoder 在这张 soft graph 上传递结构信息，最终 classifier 只读取 structure-only topology encoding，并与原始 endpoint tokens 合并，输出一个对输入顺序不敏感的 edge probability。Slot content 和 grounding 仍用于训练 generator 和构建 scaffold，但已删除独立 content-conditioning branch。训练期的辅助 losses 让 imagined neighborhoods 更真实、稳定并带有 shared-neighbor 语义；inference 最终只输出 queried pair 的二分类结果。

---

## 20. 阅读这张图时最容易混淆的三点

### Generator 不是最终任务

它生成的是供 classifier 使用的 latent neighborhood context，不是需要评估为唯一正确答案的完整图。

### Grounding candidates 不是 queried-pair batch

Pool 来自 endpoint 所属的完整 split-side feature universe，与当前 scoring batch 大小无关。

### Train-only relational head 不参与最终预测

它只负责把 topology semantics 教进 STEncoder；真正 scored output 始终来自 binary edge head。

---

## Formal implementation anchors

本文的模块边界对应当前 implementation：

- Composite / registry：src/model/egostitch/{composite,registry}.py
- Generator：src/model/egostitch/generator/egostitch.py
- Imagination / assembly：src/model/egostitch/generator/{imagine,assemble}.py
- Generator losses：src/model/egostitch/generator/losses.py
- STEncoder：src/model/egostitch/encoder/ste.py
- Classifier：src/model/egostitch/classifier/b0_v31.py
- Formal full-model config：configs/egostitch_e2e_v3_full_breadth_first.yaml
