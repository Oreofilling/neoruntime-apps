# shelf-ops — Stage 0 设备门禁（NE503 / Hailo-15H）

> 验证目标：货架商品识别在设备 ai-runtime 上到底走哪条路。
> 结论（最终）：**槽位锚定 CLIP 直分类（§7）+ yolo_world v5.4.0 逐件检测（§9）双链并存**
> —— 旧结论「检测器路径被封死（§6）」已被 §9 勘误：它只对 v5.3.0 raw-tensor 工件
> 成立，v5.4.0 的 NMS 输出语义健康且可经 RPC 流式推理。
> 详见 §7 / §9 与 [README](../README.md)。

## 背景

货架货物检测需要「图像 + 文本嵌入」双输入的开集检测模型。方案候选为
`yolo_world_v2s.hef`（HAILO15H zero-shot object detection，见
[HAILO15H_zero_shot_object_detection.rst](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO15H/HAILO15H_zero_shot_object_detection.rst)）。

设备侧未知项（本门禁要回答的）：
1. 平台注册是否接受双输入 HEF；
2. 平台是否带 yolo_world 后处理插件（决定走设备后处理 还是 Python 解码）。

验证环境：
- 设备 `192.168.93.72`，SSH TCP 隧道 `-L 19050:127.0.0.1:19050` 通到 ai-runtime 的
  socat 转发（`0.0.0.0:19050`）。
- 本地以 `PYTHONPATH=/tmp/sdkx` 引入纯 Python `neoruntime_ipc_sdk 0.3.0` SDK。
- HEF 已上传设备 `/data/aipc/models/yolo_world_v2s.hef`（27.6 MB）。

## 结论（逐项）

| # | 验证项 | 结果 |
|---|--------|------|
| 1 | `register_model` 双输入注册 | ✅ 通过（variant 必须是完整 config_json） |
| 2 | `infer_with_tensors` 双输入推理 | ✅ 通过，返回 6 个 raw 张量 |
| 3 | 输出形状 / 字节数与 parse-hef 完全一致 | ✅ 通过 |
| 4 | 嵌入（第二输入）真实影响输出 | ✅ 通过（conv 图 409513/409600 cell 随嵌入改变） |
| 5 | 图像输入真实影响输出 | ✅ 通过 |
| 6 | roundtrip 延迟 / FPS | ✅ 378–396 ms / 帧 ≈ **2.6 FPS (BS=1)** |
| 7 | 平台 yolo_world 后处理插件 | ❌ **不存在** → 走 branch B |

**总体：PASS（branch B 原始输出路径）。**

## 关键发现

### 1. 注册必须用「完整 variant + identity 后处理」

firmware 的 postproc 库（`libyolo_hailortpp_post.so`）只导出 yolov8/yolox 等固定后处理，
**没有任何 yolo_world 后处理函数**。注册时若不带 variant（或 variant 不指定
`backend_function`），推理会失败：

```
Postprocess failed: No tensor with name hailo_yolov8n_384_640/yolov8_nms_postprocess
```

这就是 runtime 自动给检测类型模型挂 yolov8 NMS 后处理、按名字找输出张量导致的。
**解法：`backend_function: "identity"`** —— 让它"不做任何后处理"，于是
`infer_with_tensors` 直接返回卷积输出的 raw 特征图：

```python
variant = json.dumps({
    "labels": [f"class_{i}" for i in range(80)],   # schema 要求必须有 labels
    "backend_function": "identity",                # 关闭后处理 -> 原始输出
})
client.register_model(
    model_path="/data/aipc/models/yolo_world_v2s.hef",
    model_id="yolo_world_v2s", owner_id="shelf-ops",
    model_type="detection", model_variant=variant,
    inputs=[
        {"name": "input_layer1", "shape": [1, 640, 640, 3], "dtype": "uint8"},
        {"name": "input_layer2", "shape": [1, 80, 512], "dtype": "uint8"},
    ],
)
```

### 2. HEF 输入/输出真值（parse-hef）

| 张量 | 作用 | shape | dtype |
|------|------|-------|-------|
| `input_layer1` | RGB 图像 | (1, 640, 640, 3) | uint8 |
| `input_layer2` | **量化后的**文本嵌入（QUANTIZED！非 float32） | (1, 80, 512) | uint8 |
| `conv48` | L80 DFL box 特征 | (80, 80, 64) | uint16 输出 |
| `normalization3` | L80 类别 logits（80 类） | (80, 80, 80) | uint16 输出 |
| `conv60` | L40 DFL box 特征 | (40, 40, 64) | uint16 |
| `normalization5` | L40 类别 logits | (40, 40, 80) | uint16 |
| `conv71` | L20 DFL box 特征 | (20, 20, 64) | uint16 |
| `normalization7` | L20 类别 logits | (20, 20, 80) | uint16 |

- 输入 `input_layer2` 是 **uint8 量化嵌入**（不在设备上做 float32 文本编码）。
- 输出经 RPC 到达 host 时是 **flat uint8 字节流**，须 `.view(np.uint16)`（native LE）
  再 reshape 成 `(L, L, C)`：
  ```python
  conv48 = outputs[0].view(np.uint16).reshape(80, 80, 64)   # DFL box 距离
  norm3  = outputs[1].view(np.uint16).reshape(80, 80, 80)   # 类别 logits
  ```
- 字节数核对（nbytes == L·L·C·2）：conv48=819200, norm3=1024000, conv60=204800,
  norm5=256000, conv71=51200, norm7=64000。

### 3. 网络确实响应两个输入（决定性敏感性实验）

固定同一张合成图像，只改 `input_layer2` 槽 0 的嵌入（zeros / 全 255 / 真实 person 编码）：

| 改变量 | conv48 (box) | normalization3 (class) |
|--------|-------------|------------------------|
| zeros → all255 | 409513 / 409600 cell 改变 | 6400 / 512000 cell 改变 |
| zeros → person | 409462 / 409600 cell 改变 | 同 all255 模式 |

- **box/DFL 图几乎全量跟随嵌入**：同一 anchor 在不同类别下回归不同 box ——
  YOLO-World 的类别相关 box 行为，第二输入真实生效。
- **class 图大部分锚点停留在量化折叠常量**（L80=8048, L40=4070, L20≈0）。
  ~~这是背景 logit 折叠，normal 行为~~ **[2026-08-21 证伪，见 §6]**：在带真实前景的
  bus.jpg 上 class 图仍折叠为同一常数 —— 不是背景效应，是 HEF 类别支路缺陷。
- 换不同图像（同样嵌入）→ conv 图全变，class 图仍折叠。

含义：Stage 2 的 Python 解码**必须把 uint16 还原成 logits 再做 sigmoid/阈值**，
并会在真实货架帧（有前景物体、80 槽全填真嵌入）上验证 dequant 标度。

### 4. 设备侧 EncodeText 可用（替代离线嵌入生成）

`client.encode_text(text)` 在设备上跑 CLIP 文本编码，返回 512 维 float 嵌入
（person/bottle 均成功，~1.9 s/次，范围约 [-0.2, 0.6]）。

- 意味着 Stage 2 的 80×512 嵌入矩阵可以**在设备上现场生成**，无需离线 torch/open_clip
  管线生成大文件随 app 打包（原计划的离线生成路径被取代）。
- 量化到 uint8 喂入 `input_layer2` 时需按 0..255 线性对齐（quant 标度在 Stage 2 校准）。

### 5. 性能

- steady-state roundtrip（BS=1，纯 RPC + Python 返回）：**378–396 ms / 帧 ≈ 2.6 FPS**。
- 低于 model-zoo 宣称的 47 FPS（那是纯 NPU 编译吞吐，不含 RPC + host 侧往返）。
- 对货架检测场景（2–5 FPS 目标）足够；app 端按 ~1 FPS 低频推理 + 事件分析运行。

## 6. 最终诊断（2026-08-21，探针 12–18 号）：类别支路缺陷，框支路完好

用 rknn_model_zoo 的 bus.jpg 已知答案（person 0.92×3 / bus 0.90 / person 0.62）
做 oracle，对浮点链路和设备 HEF 分别做了端到端验证。

**浮点链路 100% 正确**（gate_cls_probe12.py，oracle 全命中）：
tokenizer（HF CLIP ViT-B/32，seq=20 pad 49407）→ clip_text.onnx → (80,512)
单位范数嵌入；letterbox 黑边 → RGB → /255 → NCHW；输出序 [cls×3, box×3]；
cls 为 pre-sigmoid logits，box=(grid+0.5±dist)×stride。

**设备 HEF 框支路完好**（probe18）：DFL uint16 码 softmax(argmax) 解码出的
框与 oracle 五目标 IoU = 0.84 / 0.69 / 0.84 / 0.91 / 0.62。
背景锚点退化为 ~316px 巨框、目标锚点收紧为真框 → 几何过滤可用。

**设备 HEF 类别支路是信息量为零的常数场**（probe13b/14/15/16/17）：
- cls 输出 uint16 码 max=median=p99：s8=8048（512000 值仅 44 个唯一）、
  s16=4070（16 唯一）、s32=3993（7 唯一）。
- 常数对**两个输入都不变**：黑图不变；嵌入零化/全 255/增益 127.5→1600
  全部 byte-identical；换名绑定 0 差异。
- 嵌入空间排除：平台 `encode_text()` 与 HF clip_text.onnx 同空间
  （同类 cos 中位 +0.987，全部单位范数 512 维），喂平台嵌入常数不变。
- 与浮点 oracle 的 pearson ≈ +0.001（fp16 或 uint16 概率码两种解码都是）。
- 常数即 device-gate §3 观察到的 8048/4070 —— 与输入内容无关。

**结论**：`yolo_world_v2s.hef`（v5.3.0，2026-08-20 版）编译产物的
normalization3/5/7 类别输出对图像与文本输入均无响应（疑似 RepVL-PAN
文本条件 matmul 量化饱和/编译断边）。**App 侧无法通过解码修正或嵌入
对齐恢复类别分数** —— `yolo_world_decode.py` 的分数折叠到 [0.29,0.37]
是上游常数的下游症状，不是解码 bug。设备上无备用 yolo_world HEF；
`detection/hailo_yolov8n_384_640.hef` 仅 4 类（person/vehicle/face/
license_plate），无法替代开放词表。

可行修复方向（App 侧，仅用平台上完好的部件）：
DFL 框 + 几何过滤（dist∈[4,200]px 且边长 <400px）+ NMS 得候选框，再对
top-K 框用 `clip/clip_vit_b_32_image_encoder_nv12.hef` 图像编码 ×
`encode_text()` 词表嵌入余弦重打分（保留开放词表，全部在设备上）。
（该方向后经探针 33 在真实取景帧上证伪，见 §8。）

**最终取舍（2026-08-21）**：连 DFL 候选框也不要 —— 槽位多边形由排面配置
已知，检测框不提供额外信息。直接 **槽位裁剪 → CLIP 直分类**（§7）；
yolo_world 推理与候选框重打分路径全部删除。

## 7. 槽位锚定 CLIP 直分类（探针 19–32，最终架构）

对 `clip/clip_vit_b_32_image_encoder_nv12.hef` 的设备契约逐项验证，
全部 PASS：

| # | 验证项 | 结果 |
|---|--------|------|
| 1 | 注册（`model_type="embedding"` + identity variant） | ✅ 输入 spec `[1,224,224,3] uint8` |
| 2 | 推理输入 = **flat NV12**（Y 面 + UV 交织），224×224 | ✅ 75264 字节，RGB→YUV_I420→半平面拼接 |
| 3 | 输出 512 × uint8 | ✅ 带 **大偏置**：解码必须减均值再单位归一化（probe20 载荷实验） |
| 4 | 图像敏感性 | ✅ 换图输出全变；与平台 `encode_text()` 余弦排序正确 |
| 5 | 单帧延迟 | ✅ **≈ 48 ms/裁剪**；10 槽位 ≈ 0.5 s 推理 |
| 6 | `encode_text()` 文本编码 | ✅ ~2 s/prompt（一次性，缓存 `.npy` + JSON sidecar） |
| 7 | 注册稳定性 | ⚠️ 繁忙运行时 ~1/3 失败 → App 内置 3 次重试；"already exists" 良性 |

**App 集成实测（2026-08-24，10 槽位排面上线）**：一轮扫描（取帧 + 10×CLIP）
≈ 1 s（frame_seq 27→31 / 4 s）；首启 6 prompts 现场 EncodeText 一次后缓存；
10/10 槽位出码，余弦 0.266–0.319 全部高于 0.15 保持门限。

## 8. 路径 B 证伪与路径 A 整屏网格（探针 33，2026-08-24）

需求："画一个大框覆盖整个屏幕，框里计算货物"（免逐槽位标定）。候选：

- **路径 B**：复活 yolo_world **框支路**（§6 已证类别场是常数、但框支路
  对浮点 oracle 完好），仅几何过滤（dist∈[4,200]px、边长<400px）+ NMS
  得候选框，再逐框 CLIP 重打分。
- **路径 A**：一个大框 region 切 rows×cols 虚拟格子，逐格 CLIP 直分类，
  货物数 = 有货格数。

**探针 33（真实 4K 取景帧）证伪路径 B**：DFL 锚框 6400 个中 6042 个过
几何过滤，NMS 后剩 **40 个碎小框**，全部贴在背景纹理（墙沿/桌沿）上，
无一命中真实货品；同一帧浮点 oracle（x86 float 模型）出 4 个干净框、
4/4 品类正确 —— 但浮点模型上不了设备。bus.jpg 上调出的几何启发式
**不迁移**到真实取景。结论：没有类别分数，几何过滤选出的只是噪声；
路径 B 死。

**取舍（同日）**：路径 A 上线，`slot_mode: grid`：

- `config.py` 新增 `slot_mode` / `grid.region/rows/cols`（env `SLOT_MODE` /
  `GRID_ROWS` / `GRID_COLS`）；grid 模式忽略 `slots:`，由 `slots.py
  build_grid_slots()` 切虚拟格子（id `g{r}-{c}`、capacity 1、上限
  144 格，`GRID_MAX_CELLS`）。
- 格子流经**原封不动**的 §7 CLIP 扫描 / 状态机 / analytics 管线；
  `goods_summary()` 汇总 `{cells, occupied, by_code}`，`/api/state`
  增加 `mode` + `goods`。
- 渲染双胞胎：overlay.py `_draw_grid`（MJPEG 烘焙）与 app.js
  `drawGridOverlay`（HD canvas）—— thick 大框、暗色空格边线、有货格
  品类 chip、`GOODS n/N` 汇总。
- 延迟：48 ms/格串行 → 3×4=12 格 ≈ 0.6 s/轮。

sim 实测：`/api/state` `mode=grid`、`goods={cells:12, occupied:10,
by_code:{A:2,B:2,C:3,D:1,E:2}}`；浏览器侧 0 console error。

## 9. v5.4.0 平反：yolo_world 逐件检测复活（探针 49–52，2026-08-25）

> **勘误**：§头部与 §6 的「检测器路径被封死」结论**只对 v5.3.0 raw-tensor 工件
> 成立**。v5.3.0 时期判定的「zoo 检测 HEF 无法流式 / 输出是常数场」在 v5.4.0 上
> 是 dtype 错配（layer2 按 uint8 喂）造成的假象；以正确契约经 RPC 流式推理验证
> 通过。§6 原文保留不删，以本节为准。

### 9.1 真因：layer2 是 uint16，不是 uint8

HEF 元数据显示 `input_layer2` 的量化参数 qp_scale=2.7e-5、qp_zp=9778，
limvals 覆盖 CLIP ViT-B/32 文本嵌入的取值范围 —— 该层是 **uint16 [1,80,512]**：

```
u16 = clip(rint(emb / 2.7e-5 + 9778), 0, 65535)
```

v5.3.0 按 uint8 喂词表 → 运行时拒绝/空转，表象恰似「输出封死」。
改喂 uint16 后稳定流式，~0.24 s/infer。

### 9.2 输出契约（authoritative）

- 输出 = **单条 NMS-by-class float32 流**，帧 480320 B = 80×(4+300×20)。
- per class 0..79：`[f32 count][count × 5×f32 (y1,x1,y2,x2,score)]`
  —— parser 步进必须 `o += 5`，并防越界（`o + 5 > size` 即截断报错）；
  count 需校验为 0..300 内的整数。
- 片上阈值 ≈0（156 raw 框里含 0.001 级噪声）→ **主机阈值 0.25** +
  **跨类 NMS IoU 0.45**（score 降序贪心）。
- oracle：bus.jpg → person(c0) 32 框 top 0.898、bus(c5) 5 框 top 0.945；
  全黑图 → 全零。

### 9.3 注册（探针 49）

`model_type="detection"`，
`model_variant=json.dumps({"labels": ["c0".."c79"], "backend_function": "identity"})`，
显式 inputs 列表（layer1 uint8 [1,640,640,3] / layer2 uint16 [1,80,512]）；
`infer_with_tensors(inputs=[img640.flatten(), emb_u16.flatten()],
input_names=[...])` 输入-张量配对确定。注册后不 unregister（常驻）。

### 9.4 对焦根因（探针 50–52）

- probe50（App preview 帧）→ 全部 <0.15；probe51 bus.jpg 对照 person 0.897
  （词表健康）→ 根因 = **光学失焦**：Laplacian var 全帧 5.7 / 瓶区 1.4，
  vs bus.jpg 3576。
- 修复 = 设备 root ssh `aipc-cli device focus auto`（Focus -471 → -457，
  清晰度 57.0/85.8 ≈ 10–60×）。注意状态栏 "Autofocus: Off" 有误导 ——
  一次性 AF 已生效。
- 聚焦后 probe52（rtsp 原生 4K 无 overlay 帧）：bottle 0.71/0.63/0.56 +
  person 0.32，**precision 4/4，recall 3/4**（唯一漏检 = 半遮挡纸巾盒）。

### 9.5 词汇表与集成形态

- 词汇表固定 80 行 = **40 货物 prompt + 40 负样本**（person/hand/shelf/…）；
  CLIP ViT-B/32 文本嵌入离线预计算（HF `openai/clip-vit-base-patch32`
  tokenizer + clip_text.onnx，SEQ_LEN 20、PAD 49407）→ (80,512) f32。
  随包资产 `assets/yolo_world_vocab.npy` + `.json` sidecar（出处与验证
  日期记录在 sidecar provenance 字段）。**不是** tools/generate_embeddings.py
  的 open_clip laion2b 权重，也不走设备 encode_text —— 只用 probe 验证过的矩阵。
- 产品化 `detector.py`（`GoodsDetector`：letterbox → infer → parse → 阈值 →
  跨类 NMS → un-letterbox → 格映射）；app.py 每 tick 在网格 CLIP 之外叠
  检测框，新计数 **`ITEMS m`** = 中心落在 grid region 内的去重货物框数；
  `GOODS n/N`（格 CLIP）语义不动。负样本命中（person 等）画灰框、不计入。
- 失败语义：注册/加载失败 → `detector=None`（CLIP 照跑，`/api/health`
  `detector:"unavailable"`）；单 tick 推理失败 → 保留上次框 ≤3 tick 后清空
  （`"degraded"`，不污染 CLIP 的 degraded 计数）。回滚：`DETECT_ENABLED=0`。

## 分支决策（历史沿革）

1. **branch B**（原始输出 + Python 解码）：注册 OK 但无 yolo_world 后处理
   → `yolo_world_decode.py`。**已被 §6 证伪** —— 类别支路是常数场，
   解码无法恢复。
2. **branch C**（yolov8n 兜底）：平台 `yolov8n` 头仅 person/vehicle/face/
   license_plate 4 类，看不见货架商品 → 不可用。
3. **最终**：槽位锚定 CLIP 直分类（§7）—— 不依赖任何检测器，
   `clip_vit_b_32_image_encoder_nv12.hef` 图像塔 + `encode_text()` 词表，
   全部在设备上，开放词表能力保留。
4. **路径 B**（§6 框支路 + 几何过滤 + CLIP 重打分，2026-08-24 提案）：
   探针 33 证伪 —— 真实 4K 帧上几何过滤后只剩 40 个背景噪声碎框（§8）。
5. **整屏网格计数（路径 A）**：`slot_mode: grid` 免逐槽位标定的计数
   模式，切虚拟格子复用 §7 管线（§8）。
6. **逐件检测复活（§9，2026-08-25）**：yolo_world **v5.4.0** NMS 输出 +
   uint16 词表契约 → chain A 逐件框 + `ITEMS m`，与 §7/§8 的 CLIP 计数
   双链并存；v5.3.0「封死」结论系 dtype 错配假象（§9.1 勘误）。

## 复现

探针脚本（本机 + SSH 隧道）：
- `gate_raw_probe.py`    — 注册 variant A/B/C/D 四连测，证明仅 D(identity) 成功。
- `gate_raw_verify.py`   — D 注册 → 6 raw 张量形状/字节数核对 + FPS 循环。
- `gate_encode_probe.py` — EncodeText 设备侧探针 + 真实嵌入 vs 假字节的 class 图对比。
- `gate_diff2.py`        — 决定性敏感性实验（定图变嵌入 / 定嵌入变图）。
- `gate_cls_probe2..18`  — §6 诊断链：浮点 oracle 全对 → 设备框支路完好
  （IoU 0.62–0.91）→ 类别支路常数场（对图/嵌入均无响应）。
- `gate_cls_probe19..32` — §7 CLIP 契约链：注册 spec / NV12 布局 / uint8
  解码（probe20 载荷实验）/ 48 ms 计时 / EncodeText 词表 / App 集成。
- `gate_cls_probe33`    — §8 路径 B 证伪：真实 4K 帧 DFL 几何过滤 →
  40 个噪声碎框 vs 浮点 oracle 4 个干净框。
- `gate_cls_probe49..52` — §9 v5.4.0 契约链：uint16 注册/流式推理、
  bus.jpg oracle、对焦根因定位、聚焦实帧终审。

运行方式：`PYTHONPATH=/tmp/sdkx python3 gate_cls_probeN.py`
（先建隧道：`ssh -L 19050:127.0.0.1:19050 root@192.168.93.72`）。

## 日期 / 环境

- 验证日期：2026-08-20（branch B）→ 2026-08-21（§6 诊断 + §7 契约）→
  2026-08-24（App 集成实测上线 + §8 路径 B 证伪 / 路径 A 网格计数）→
  2026-08-25（§9 v5.4.0 平反 + 逐件检测集成）
- 设备：NE503 (192.168.93.72)，ai-runtime 经 socat 转发 19050
- SDK：neoruntime_ipc_sdk 0.3.0（探针）/ 0.4.0（App，注意注册 kwarg 是
  `model_variant` 而非 `variant`）
- HEF：`yolo_world_v2s.hef` v5.3.0（弃用）；`clip_vit_b_32_image_encoder_nv12.hef`（现役）