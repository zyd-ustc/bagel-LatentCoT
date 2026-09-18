# Bagel 上 FlowEdit 的 zero-shot 实现

基于仓库 `ByteDance-Seed/Bagel` 现有接口：`inferencer.py` 的 `InterleaveInferencer`，以及 `modeling/bagel/bagel.py` 的 `prepare_*`、`forward_cache_update_*`、`generate_image`、`_forward_flow`。不新增模型模块，不改 MoT / 训练目标。权重保持官方 checkpoint。

FlowEdit 原文（Kulikov et al., ICCV 2025）只提供积分器，速度场仍用 Bagel 已训练的 \(v_\theta(x,t\mid \text{prefix KV})\)。

---

## 1. 官方推理里速度场是什么

`generate_image`（`bagel.py`）对 packed VAE token 做 Euler：

```text
x_t = packed_init_noises
timesteps = linspace(1, 0, num_timesteps)
timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
dts = timesteps[:-1] - timesteps[1:]
timesteps = timesteps[:-1]

for i, t in enumerate(timesteps):
    v_t = _forward_flow(x_t, t, past_key_values, cfg_*_past_key_values, ...)
    x_t = x_t - v_t * dts[i]    # 源码注释：velocity pointing from data to noise
```

`_forward_flow` 每步：

1. `packed_sequence` 长度 = `sum(packed_seqlens)`，在 `packed_text_indexes` 写入 `start_of_image` / `end_of_image` 的 embedding，在 `packed_vae_token_indexes` 写入 `vae2llm(x_t) + time_embedder(t) + latent_pos_embed`。
2. `language_model.forward_inference(..., past_key_values=..., update_past_key_values=False, is_causal=False)`。
3. `v_t = llm2vae(output)[packed_vae_token_indexes]`。
4. 若 `cfg_text_scale > 1` 或 `cfg_img_scale > 1`，用另外两套 cache 再算 `cfg_text_v_t` / `cfg_img_v_t`，按已有公式合成，再做 `cfg_renorm_type`。

前缀（prompt、可选的 clean VAE / ViT）只存在于 `past_key_values`。积分过程中 `update_past_key_values=False`，条件不会随 `x_t` 改写。这是 FlowEdit 能直接接的原因：同一套 `_forward_flow`，换 cache 就能换条件。

`x_t` 的布局与前缀长度无关，形状是

```text
(num_image_tokens, latent_channel * latent_patch_size ** 2)
num_image_tokens = (H / latent_downsample) * (W / latent_downsample)
```

`prepare_vae_latent` 在 query 侧固定为 `[start_of_image] + VAE tokens + [end_of_image]`，即 `packed_seqlens = num_image_tokens + 2`。两套不同长度的文本前缀可以共用同一份 `x_t`，不能共用同一份 `packed_indexes` / `packed_key_value_indexes`。

---

## 2. 官方 CFG cache 怎么切（编辑 vs T2I）

`interleave_inference` 维护三个 `gen_context`，结构相同：

```text
gen_context = {
    'kv_lens': [int],
    'ropes': [int],
    'past_key_values': NaiveCache(num_hidden_layers),
}
```

写入前缀只用现有三个方法：

| 方法 | 底层 |
|---|---|
| `update_context_text(text, ctx)` | `prepare_prompts` → `forward_cache_update_text` |
| `update_context_image(image, ctx, vae=True, vit=True)` | `prepare_vae_images` → `forward_cache_update_vae`；`prepare_vit_images` → `forward_cache_update_vit` |

输入列表按顺序扫描时，cache 切分是：

- 遇到 `str`：先 `cfg_text_context = deepcopy(gen_context)`（尚未写入这段文本），再把文本写入 `gen_context` 和 `cfg_img_context`。
- 遇到 `Image`：只写入 `gen_context`（`vae=not understanding_output`），然后 `cfg_text_context = deepcopy(gen_context)`。`cfg_img_context` 不写图。

因此常见两种前缀：

**T2I**，`input_lists = [prompt]`

- `gen_context`：prompt
- `cfg_text_context`：空
- `cfg_img_context`：prompt（T2I 下通常 `cfg_img_scale=1`，这套不参与）

**编辑**，`input_lists = [source_image, instruction]`

- `gen_context`：source 的 VAE + ViT + instruction
- `cfg_text_context`：source 的 VAE + ViT（无 instruction）
- `cfg_img_context`：instruction（无 source 图）

`gen_image` 对三套 context 分别调用 `prepare_vae_latent` / `prepare_vae_latent_cfg`，把 query 索引接到各自的 `kv_lens` / `ropes` 上，再一次性交给 `generate_image`。

FlowEdit 不使用「中途改 `gen_context` 里的 prompt」这条路径（实验已证 zero-shot 无效）。它使用两套完整前缀，每步各算一个 `v_t`，用差速积分第三份 latent。

---

## 3. FlowEdit 差速接到 Bagel 的 Euler

FlowEdit 在源条件 / 目标条件上各评一次速度，积分编辑 latent。对应 Bagel 的符号：

- \(Z_0^{\mathrm{src}}\)：源图在 VAE latent、且已按 `decode_image` 的逆过程 pack 成与 `packed_init_noises` 同形的张量，记为 `x_src0`
- \(Z_t^{\mathrm{src}} = (1-t)\,x_{\mathrm{src0}} + t\,\varepsilon_t\)
- \(Z_t^{\mathrm{tar}} = x_{\mathrm{edit}} + Z_t^{\mathrm{src}} - x_{\mathrm{src0}}\)
- \(v^{\mathrm{src}} = V_\theta(Z_t^{\mathrm{src}}, t \mid \text{src caches})\)
- \(v^{\mathrm{tar}} = V_\theta(Z_t^{\mathrm{tar}}, t \mid \text{tar caches})\)
- \(v^\Delta = v^{\mathrm{tar}} - v^{\mathrm{src}}\)

Bagel 的 Euler 是 `x ← x - v * dt`（\(v\) 从 data 指向 noise，\(t:1\to 0\)）。差速用同一套符号：

```text
x_edit = x_edit - (v_tar - v_src) * dts[i]
```

\(t\) 与 `dts` 必须与 `generate_image` 里那段 `timestep_shift` 变换完全一致，否则和训练时的 `time_embedder` 输入对不上。

`n_avg`（FlowEdit 对多个 \(\varepsilon_t\) 平均 \(v^\Delta\)）在实现上就是对同一个 `t` 抽 `n_avg` 个 `torch.randn_like(x_src0)`，分别构造 `z_src` / `z_tar`，把 `(v_tar - v_src)` 平均后再乘 `dts[i]`。

编辑只作用在时间窗 `[n_min, n_max]`（按 **变换后的 \(t\)**，不是 step 下标）。窗外两选一，都用现有积分器：

- `t > n_max`（更接近噪声）：对 `x_edit` 用 **src** 的 `_forward_flow` 走，保持与源切片一致；或直接跳过（`x_edit` 保持 `x_src0`）。推荐前者，与从噪声出发的官方 T2I 更接近。
- `t < n_min`（更接近数据）：对 `x_edit` 用 **tar** 的 `_forward_flow` 走完，让高频按目标条件收敛。

---

## 4. 两套前缀怎么建

只调用 `init_gen_context` / `update_context_text` / `update_context_image`。下面两种输入对应两种实验，不要混用 cache 语义。

### 4.1 文本条件对（对应 FlowEdit 论文的 \(c_{\mathrm{src}}, c_{\mathrm{tar}}\)）

源可以是一张真实图，也可以是已经用 `gen_image` 在 `c_src` 下生成的图。前缀里 **不写 source 图**（图只作为 `x_src0`）。

```text
src_ctx          = update_context_text(c_src, init_gen_context())
src_cfg_text_ctx = init_gen_context()                          # 与 T2I 的 cfg_text 相同：空前缀
src_cfg_img_ctx  = deepcopy(src_ctx)                           # T2I 下 cfg_img_scale=1，可占位

tar_ctx          = update_context_text(c_tar, init_gen_context())
tar_cfg_text_ctx = init_gen_context()
tar_cfg_img_ctx  = deepcopy(tar_ctx)
```

`_forward_flow` 两侧都设 `cfg_text_scale` 为 T2I 常用值（README：4.0–8.0），`cfg_img_scale=1.0`。这样每侧内部仍走官方 text CFG，差速是「两条 CFG 之后的 \(v\)」相减。

### 4.2 官方编辑前缀对（source 图进 KV）

若要把官方编辑器当成 \(V(\cdot\mid \text{image},\text{text})\) 来做差速：

```text
src_ctx = init_gen_context()
src_ctx = update_context_image(source_image, src_ctx, vae=True, vit=True)
src_ctx = update_context_text(c_src, src_ctx)          # c_src 用源图描述，或空字符串

tar_ctx = init_gen_context()
tar_ctx = update_context_image(source_image, tar_ctx, vae=True, vit=True)
tar_ctx = update_context_text(c_tar, tar_ctx)          # c_tar 用编辑指令
```

两侧的 `cfg_text_context` / `cfg_img_context` 按第 2 节官方规则切：先图后文，使

- 全条件 = VAE + ViT + 文本
- `cfg_text` = VAE + ViT
- `cfg_img` = 文本

`cfg_text_scale`、`cfg_img_scale` 用 `app.py` 编辑默认（`4.0` / `2.0`），`cfg_renorm_type="text_channel"`。

注意：4.2 里 clean source 已经在 KV 里，同时又用 `x_src0` 做平行四边形。这是「官方编辑条件 + FlowEdit 积分器」，不是论文原设定。4.1 才是论文原设定在 Bagel T2I 速度场上的直接移植。建议先跑 4.1。

---

## 5. `x_src0`：`decode_image` 的逆过程

`InterleaveInferencer.decode_image`：

```text
(h, w) = (H / latent_downsample, W / latent_downsample)
latent: (num_image_tokens, C * p * p)
    -> (1, h, w, p, p, C)
    -> einsum "nhwpqc->nchpwq"
    -> (1, C, h*p, w*p)
    -> vae_model.decode
```

源图进积分器时做相反操作，VAE 与 `decode_image` 为同一个 `self.vae_model`，图像先走 `self.vae_transform`（与 `update_context_image` 的 VAE 分支同一套）：

```text
img = vae_transform.resize_transform(pil_img2rgb(source_image))
img_tensor = vae_transform(img)                    # 与 prepare_vae_images 内部 transforms(image) 一致
z = vae_model.encode(img_tensor)                   # (1, C, h*p, w*p)，按 modeling/autoencoder.py 的实际返回取值
z = z.reshape(1, C, h, p, w, p)
z = einsum "nchpwq->nhwpqc"
x_src0 = z.reshape(h * w, p * p * C)
```

`H, W` 必须与随后 `prepare_vae_latent(..., image_sizes=[(H, W)])` 一致，否则 `packed_vae_position_ids` 和 token 数对不上。编辑时 `image_shapes` 用 `vae_transform.resize_transform` 之后的 `size[::-1]`，与 `interleave_inference` 里对 `Image` 的处理相同。

`x_edit` 初值：`x_src0.clone()`（FlowEdit 的 \(Z^{\mathrm{FE}}\) 从源图出发，不从 `packed_init_noises` 出发）。

---

## 6. 每侧 query 索引必须按各自 `kv_lens` 重算

`prepare_vae_latent(curr_kvlens, curr_rope, image_sizes, new_token_ids)` 产出主路径的

```text
packed_text_ids, packed_text_indexes,
packed_init_noises, packed_vae_position_ids, packed_vae_token_indexes,
packed_seqlens, packed_position_ids, packed_indexes,
key_values_lens, packed_key_value_indexes
```

`prepare_vae_latent_cfg(curr_kvlens, curr_rope, image_sizes)` 产出

```text
cfg_packed_position_ids, cfg_packed_query_indexes,
cfg_key_values_lens, cfg_packed_key_value_indexes
```

src / tar 的 `kv_lens`、`ropes` 一般不等（文本长度不同）。因此：

```text
gen_in_src      = model.prepare_vae_latent(src_ctx['kv_lens'], src_ctx['ropes'], [image_shape], new_token_ids)
gen_in_src_ct   = model.prepare_vae_latent_cfg(src_cfg_text_ctx['kv_lens'], src_cfg_text_ctx['ropes'], [image_shape])
gen_in_src_ci   = model.prepare_vae_latent_cfg(src_cfg_img_ctx['kv_lens'], src_cfg_img_ctx['ropes'], [image_shape])

gen_in_tar      = model.prepare_vae_latent(tar_ctx['kv_lens'], tar_ctx['ropes'], [image_shape], new_token_ids)
gen_in_tar_ct   = model.prepare_vae_latent_cfg(tar_cfg_text_ctx['kv_lens'], tar_cfg_text_ctx['ropes'], [image_shape])
gen_in_tar_ci   = model.prepare_vae_latent_cfg(tar_cfg_img_ctx['kv_lens'], tar_cfg_img_ctx['ropes'], [image_shape])
```

共用的只有 `x_src0` / `x_edit` 和 `packed_vae_position_ids` 的空间网格（同 `image_shape`）。`gen_in_src['packed_init_noises']` 丢弃，不参与初值。

`packed_vae_token_indexes` 在 query 坐标系里，两侧都是 `[1 .. num_image_tokens]`，可以断言相等。

---

## 7. 建议挂载位置与调用结构

不改 `Bagel._forward_flow`。在 `InterleaveInferencer` 旁增加一个方法，内部复制 `generate_image` 的时间表，两次调用已有 `_forward_flow`。推荐文件：`inferencer.py`，与 `gen_image` 并列。

伪代码（名称仅作局部变量，对应源码符号）：

```python
@torch.no_grad()
def gen_image_flowedit(
    self,
    image_shape,
    x_src0,
    src_ctx, src_cfg_text_ctx, src_cfg_img_ctx,
    tar_ctx, tar_cfg_text_ctx, tar_cfg_img_ctx,
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=(0.4, 1.0),
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
    num_timesteps=50,
    timestep_shift=3.0,
    n_min=0.0,
    n_max=1.0,
    n_avg=1,
):
    gen_in_src = self.model.prepare_vae_latent(
        src_ctx['kv_lens'], src_ctx['ropes'], [image_shape], self.new_token_ids)
    gen_in_tar = self.model.prepare_vae_latent(
        tar_ctx['kv_lens'], tar_ctx['ropes'], [image_shape], self.new_token_ids)
    # 两侧再各 prepare_vae_latent_cfg × 2，同 gen_image

    x_edit = x_src0.clone()
    timesteps = torch.linspace(1, 0, num_timesteps, device=x_edit.device)
    timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
    dts = timesteps[:-1] - timesteps[1:]
    timesteps = timesteps[:-1]

    for i, t in enumerate(timesteps):
        timestep = torch.tensor([t] * x_edit.shape[0], device=x_edit.device)

        if t > cfg_interval[0] and t <= cfg_interval[1]:
            text_scale, img_scale = cfg_text_scale, cfg_img_scale
        else:
            text_scale, img_scale = 1.0, 1.0

        if t > n_max:
            v = self.model._forward_flow(
                x_t=x_edit, timestep=timestep,
                past_key_values=src_ctx['past_key_values'],
                cfg_text_past_key_values=src_cfg_text_ctx['past_key_values'],
                cfg_img_past_key_values=src_cfg_img_ctx['past_key_values'],
                cfg_text_scale=text_scale, cfg_img_scale=img_scale,
                cfg_renorm_min=cfg_renorm_min, cfg_renorm_type=cfg_renorm_type,
                **索引参数_src,
            )
            x_edit = x_edit - v * dts[i]
            continue

        if t < n_min:
            v = self.model._forward_flow(
                x_t=x_edit, timestep=timestep,
                past_key_values=tar_ctx['past_key_values'],
                ...  # tar 的 cfg cache 与索引
            )
            x_edit = x_edit - v * dts[i]
            continue

        v_delta_acc = 0
        for _ in range(n_avg):
            eps = torch.randn_like(x_src0)
            z_src = (1.0 - t) * x_src0 + t * eps
            z_tar = x_edit + z_src - x_src0
            v_src = self.model._forward_flow(x_t=z_src, timestep=timestep, ... src ...)
            v_tar = self.model._forward_flow(x_t=z_tar, timestep=timestep, ... tar ...)
            v_delta_acc = v_delta_acc + (v_tar - v_src)
        x_edit = x_edit - (v_delta_acc / n_avg) * dts[i]

    return self.decode_image(x_edit, image_shape)
```

`_forward_flow` 的 `timestep` 在源码里写成 `torch.LongTensor`，实际传入的是 `torch.tensor([t] * x_t.shape[0])`、`t` 为 float。跟官方 `generate_image` 保持同一写法。

`enable_taylorseer` 第一版关掉。TaylorSeer 缓存在 `language_model.model.cache_dic` 上是单份的，双分支会串缓存。

`cfg_type` 保持默认 `"parallel"`。

---

## 8. 与 `gen_image` 的参数对应

| `generate_image` / `gen_image` | FlowEdit 循环 |
|---|---|
| `past_key_values` | 每步两套：`src_ctx` / `tar_ctx` |
| `cfg_text_past_key_values` | 每套前缀各自的 `cfg_text_context` |
| `cfg_img_past_key_values` | 每套前缀各自的 `cfg_img_context` |
| `packed_init_noises` | 不用于初值；`x_edit` 从 `x_src0` 起步 |
| `num_timesteps`, `timestep_shift` | 原样 |
| `cfg_interval`, `cfg_renorm_*`, `cfg_*_scale` | 原样，作用在单侧 `_forward_flow` 内部 |
| `update_past_key_values` | 保持 `False` |
| `is_causal` | 保持 `False` |

新增的只是积分器外的三个标量：`n_min`、`n_max`、`n_avg`。建议初值与 FlowEdit 在 SD3/FLUX 上的习惯同量纲（变换后的 \(t\in[0,1]\)）：先 `n_max=0.8`、`n_min=0.2`、`n_avg=1`，再扫窗。

---

## 9. 入口：不要走 `interleave_inference` 的单前缀循环

`interleave_inference` 把所有输入写入同一套 `gen_context` 再调一次 `gen_image`。FlowEdit 需要两套 context 并存，因此新入口自己组 cache，不要复用 `interleave_inference`。

建议的调用顺序：

1. 读入 `source_image`, `c_src`, `c_tar`。
2. `source_image = vae_transform.resize_transform(pil_img2rgb(source_image))`，`image_shape = source_image.size[::-1]`。
3. 按第 5 节得 `x_src0`。
4. 按第 4.1 或 4.2 节得六套 context（src/tar × 主/cfg_text/cfg_img）。
5. 调第 7 节的循环，`decode_image`。

对照实验（证明差速而不是换字在起作用）用同一 `x_src0`、同一 `num_timesteps` / `timestep_shift`：

- 对照 1：只建 `tar_ctx`，`x_t = packed_init_noises`，走官方 `gen_image`（普通 T2I / 官方编辑）。
- 对照 2：只建 `tar_ctx`，`x_t` 从 `x_src0` 起步但积分用单侧 `v_tar`（img2img 式，没有 \(v_{\mathrm{tar}}-v_{\mathrm{src}}\)）。
- 对照 3：本文循环。

对照 2 与对照 3 的差才是 FlowEdit 积分器本身。

---

## 10. 实现时不要动的部分

- `Qwen2MoTDecoderLayer` 路由、`mode="gen"`、`packed_vae_token_indexes` 硬路由。`_forward_flow` 已设 `extra_inputs["mode"]="gen"`，双分支原样传。
- `forward_cache_update_*` 的因果预填。两套前缀各自预填一次即可，积分中不再调用。
- 训练代码、`edit_dataset.py` 的 `[source VAE+ViT][instruction][noised target]` 序列。本方案 zero-shot，不改数据。
- `packed_sequence` 里那两个 `start_of_image` / `end_of_image` query token。不要在这里插入额外文本。

每步代价：编辑窗内是官方一步的 \(2\times n_{\mathrm{avg}}\) 倍（再乘单侧 CFG 的 2 或 3 次 `forward_inference`）。`n_avg=1`、`cfg_img_scale=1` 时，编辑窗内约为官方 T2I 一步的 4 倍（src/tar × cond/uncond）。

---

## 11. 预期与失败模式（用来读结果，不是改代码）

- `n_max` 太小：差速开始太晚，低频已在源切片上锁死，只会动纹理。
- `n_min=0` 且 `cfg_text_scale` 很大：后期仍用差速，容易把 `x_edit` 推离 VAE 流形，decode 发糊。后期切到单侧 `v_tar`（第 3 节窗外规则）是为了这个。
- 4.2 前缀 + 大 `cfg_img_scale`：clean source 已在 KV 里，差速再叠平行四边形，身份过锁、指令无效果。先 4.1。
- `c_src` 与源图不符：\(v_{\mathrm{src}}\) 不是源切片上的场，差速无意义。真实图必须有匹配的源描述，或先用官方理解接口给源图出 caption 再作为 `c_src`（`understanding_output=True` 的 `gen_text`）。

本方案验证的问题只有一个：在 **不改 `packed_init_noises` 起源、不改已写入的 src 前缀、不训练** 的前提下，用官方 `_forward_flow` 的差速，能否把 `x_src0` 运到 `c_tar` 的语义切片。若件数 / 槽位仍不动，说明 \(V_\theta(\cdot\mid c)\) 在相邻条件间不够局部光滑，zero-shot 运输到此为止。
