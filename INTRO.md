# CS336 Assignment 1 (basics) 作业说明

本文档按 PDF 章节顺序概括作业内容，便于查阅。

---

## 1. Assignment Overview（作业概览）

目标：从零实现训练标准 Transformer 语言模型所需的组件，并完成训练与实验。

需要实现的四块：BPE 分词器（§2）、Transformer LM（§3）、交叉熵与 AdamW（§4）、训练循环与序列化/加载（§5）。

需要跑的五件事：在 TinyStories 上训练 BPE；用该 BPE 把数据转成 token ID；在 TinyStories 上训练 Transformer LM；用训练好的 LM 做生成并算 perplexity；在 OpenWebText 上训练并提交 leaderboard。

允许的 PyTorch：仅 nn.Parameter、容器类（Module/ModuleList/Sequential 等）、optim.Optimizer 基类。禁止用 nn.Linear、F.linear、现成 Adam 等。

AI 政策：允许用 LLM 问概念或底层编程问题，禁止直接解题；建议关掉 AI 自动补全。

代码结构：实现写在 cs336_basics/；adapters.py 里只写“胶水”调用；测试在 test_*.py，通过适配器调用你的代码，不要改测试文件。

提交：Gradescope 交 writeup.pdf 与 code.zip；Leaderboard 按指定仓库 README 提 PR。

数据：TinyStories 与 OpenWebText 均为单一大文本；课程机器在 /data；本地见 README。

蓝框（Low-Resource Tip）：作业中多处蓝框给出在 Apple Silicon/CPU、小数据、少步数下的建议（如 M3 Max 约 5 分钟 MPS、约 30 分钟 CPU 可训出可读文本）。

---

## 2. Byte-Pair Encoding (BPE) Tokenizer（§2）

### 2.1 The Unicode Standard

Unicode 将字符映射到码点（整数）；ord/chr 在 Python 中做字符与整数转换。书面题 unicode1 要求：chr(0) 是什么、其 __repr__ 与 print 的区别、该字符在文本中的表现（各一句话）。

### 2.2 Unicode Encodings

用编码（如 UTF-8）把字符变成字节序列，便于在 0–255 上做 BPE，避免 15 万级码点词表。书面题 unicode2：为何偏好 UTF-8；一个错误逐字节 decode 函数的反例；一个无法解码的两字节序列示例。

### 2.3 Subword Tokenization

字节级太长；词级 OOV 严重。子词折中：词表稍大、序列更短。BPE 通过迭代合并最高频字节对来扩词表。

### 2.4 BPE Tokenizer Training

三步：词表初始化（256 字节 + special tokens）；预分词（GPT-2 正则，用 re.finditer，按 special tokens 先切再预分词，不跨文档合并）；计算 merges（同频按字典序取大，不跨预分词边界）。Special tokens 如 <|endoftext|> 单独加入词表。可并行预分词（按 special token 切 chunk）；merge 步可做增量 pair 计数加速。

例题（bpe_example）：给定 corpus 与 special token，演示词表、预分词计数、多轮 merge 的顺序。

### 2.5 Experimenting with BPE Tokenizer Training

建议先用小数据/验证集做“debug dataset”；用 cProfile/scalene 找瓶颈。书面题：TinyStories 10K vocab 训练时间/内存、最长 token、profiling 最耗时部分；OpenWebText 32K 最长 token、与 TinyStories tokenizer 对比。

### 2.6 BPE Tokenizer: Encoding and Decoding

编码：预分词 -> 每预分词转 UTF-8 字节 -> 按 merges 顺序在预分词内合并 -> 查词表得 ID。Special tokens 不拆；大文件需按 chunk 且不跨 chunk 断 token。解码：ID 查词表得字节、拼接、UTF-8 解码；非法字节用 U+FFFD（errors='replace'）。

书面题 tokenizer：实现 Tokenizer 类（__init__、from_files、encode、encode_iterable、decode）并通过 adapters.get_tokenizer。

### 2.7 Experiments

书面题 tokenizer_experiments：采样 10 文档、两种 tokenizer 的压缩比；用 TinyStories tokenizer 编 OWT 样本的现象；吞吐与 825GB Pile 预估时间；用 uint16 存 token 的原因。

---

## 3. Transformer Language Model Architecture（§3）

### 3.1–3.2 整体与输出

LM 输入 token ID，经 embedding、num_layers 个 Transformer block、最后 RMSNorm、再线性得到 (batch, seq_len, vocab_size) 的 logits。Pre-norm：每子层先 norm 再算、再加残差；最后再一层 RMSNorm。

### 3.3 Batching、Einsum、内存顺序

Batch/序列/头等维度可统一当“批维”处理；推荐用 einsum/einops 写矩阵运算，便于读 shape。数学用列向量，PyTorch 用行主序；若用 einsum 可不必纠结转置。

### 3.4 Basic Building Blocks

初始化：Linear 用 N(0, 2/(din+dout)) 截断 [−3σ,3σ]；Embedding 用 N(0,1) 截断 [−3,3]；RMSNorm 的 gain 初始为 1。用 trunc_normal_。

Linear：y = Wx，无 bias，存 W 为 (out_features, in_features)。Embedding：查表 (vocab_size, d_model)。

### 3.5 Pre-Norm Transformer Block

RMSNorm：式 (4)，先转 float32 再算再转回原 dtype。SwiGLU 前馈：式 (7)，d_ff ≈ (8/3)*d_model 且为 64 倍数。RoPE：式 (8)(9)，对 Q/K 按位置旋转，可用 buffer 存 cos/sin。Scaled dot-product attention：式 (11)，mask 处填 -inf；支持任意 batch 维。Causal multi-head self-attention：Q/K/V 投影、RoPE 只加在 Q/K、causal mask、再 O 投影；dk=dv=d_model/num_heads。

Block 公式：y = x + MHA(RMSNorm(x))；z = y + FFN(RMSNorm(y))。

### 3.6 The Full Transformer LM

把 embedding、num_layers 个 block、最后 RMSNorm、lm_head 拼起来。书面题 transformer_accounting：GPT-2 XL 参数量与显存；前向所有矩阵乘及 FLOPs；各部件占比；small/medium/large 对比；context 从 1024 到 16384 对 FLOPs 的影响。

---

## 4. Training a Transformer LM（§4）

### 4.1 Cross-entropy loss

式 (16)(17)：对每个位置算 −log p(xi+1|x1:i)；实现时减 max 稳数值，log-sum-exp 可消去 log/exp。Perplexity 为式 (18)。

### 4.2 The SGD Optimizer

式 (19)：θ ← θ − α∇L。示例给的是带 1/√(t+1) 衰减的 SGD，并演示如何继承 Optimizer、用 param_groups 和 state。书面题 learning_rate_tuning：不同 lr 下 loss 行为。

### 4.3 AdamW

按 Loshchilov & Hutter 算法 2：一阶/二阶矩、bias 校正、先更新参数再减权 λθ。书面题 adamwAccounting：峰值显存分解、GPT-2 XL 最大 batch、一步 FLOPs、MFU 与 400K 步训练时间。

### 4.4 Learning rate scheduling

Cosine + warmup：t<Tw 线性升到 αmax；Tw≤t≤Tc 按余弦降到 αmin；t>Tc 恒为 αmin。实现 get_lr_cosine_schedule。

### 4.5 Gradient clipping

对全部参数梯度的 L2 范数，若超过 M 则整体缩放为 M（加小 ε 防除零），原地修改 grad。

---

## 5. Training loop（§5）

### 5.1 Data Loader

数据为整段 token 序列；每次随机取起点，得到 (inputs, targets)，均为 (batch_size, context_length)。支持从 memmap 的数组采样；device 用 'cpu' 或 'cuda:0'/'mps'。

### 5.2 Checkpointing

save_checkpoint：存 model.state_dict()、optimizer.state_dict()、iteration。load_checkpoint：读回并 load_state_dict，返回 iteration。

### 5.3 Training loop

书面题 training_together：写训练脚本，支持超参配置、memmap 加载、定期 checkpoint、记录 train/val（可接 wandb）。

---

## 6. Generating text（§6）

从 prompt 开始，每步用 LM 最后一位置 logits，经 temperature softmax 与（可选）top-p 采样得下一 token，直到 EOS 或达到 max_tokens。书面题 decoding：实现上述解码，支持 temperature 与 top-p。

---

## 7. Experiments（§7）

### 7.1 实验与交付

用小模型（约 17M 参数）和 TinyStories 快速试；系统做消融与超参；用 experiment_log 记录步骤与时间。书面题 experiment_log：建好日志与曲线（step + wallclock）。

### 7.2 TinyStories

给定一组默认超参（vocab 10K、context 256、d_model 512、4 层 16 头、总 token 约 327.68M 等）；需自己调学习率、warmup、AdamW 参数、weight decay。书面题 learning_rate：学习率 sweep、验证 loss≤1.45、发散 lr 与“edge of stability”。batch_size_experiment：不同 batch 的曲线与结论。generate：至少 256 token 样例与流畅度、影响质量的两点。

### 7.3 Ablations and architecture modification

layer_norm_ablation：去掉 RMSNorm 训练，看是否要更小 lr。pre_norm_ablation：改成 post-norm 对比。no_pos_emb：去掉 RoPE（NoPE）对比。swiglu_ablation：SwiGLU vs 参数量近似的 SiLU FFN。

### 7.4 Running on OpenWebText

OWT 更真实、更杂。main_experiment：同架构、同迭代在 OWT 上训练；交曲线、与 TinyStories 的 loss 对比、生成样例与质量分析。

### 7.5 Your own modification + leaderboard

规则：最多 1.5 H100 小时、仅用提供的 OWT 训练数据；其余自由。书面题 leaderboard：在 1.5 小时内尽量压低验证 loss，交最终 loss、带 wallclock 的曲线、做法说明，并提交至指定 GitHub；需优于 loss 5.0 的 baseline。

---

## References（参考文献）

PDF 末尾列出 BPE、Transformer、RMSNorm、RoPE、SwiGLU、AdamW、nucleus sampling、LLaMA、Qwen 等文献，实现与书面题可直接对照相应章节。
