# ShelfOps — 货架槽位识别（Hailo-15H / NE503）

货架槽位识别 showcase：**CLIP 计数**（槽位锚定直分类或整屏网格，
`GOODS n/N`）叠加 **yolo_world v5.4.0 逐件检测**（`ITEMS m` + 每件
label/score/格号框，见[计数模式](#计数模式槽位-vs-整屏网格)）。
历史上设备检测路径曾被证伪（v5.3.0 raw 契约：类输出为信息-free 常数，
[docs/device-gate.md](docs/device-gate.md) §6–§8；平台 `yolov8n` 头只有
person/vehicle/face），v5.4.0 经 uint16 双输入契约**平反**（§9）——
检测框与 CLIP 格计数双链并存，CLIP 仍锚定在**槽位/网格多边形**上：

```
帧 → 每个槽位多边形裁剪（10 块）→ CLIP ViT-B/32 图像编码（~48 ms/块）
   → 与词表文本嵌入做余弦 → 每槽位 (品类 | 空位, cos) → 库存状态机
   → 缺货/补货事件 + 时间桶热力图
```

运行于 NeoRuntime / NE503 平台，与 gym-ops 同构：Flask + hailo_ipc_sdk +
app.yaml + Docker + 单元测试 + 自动 bundle CI。

## 功能一览

- **每槽位品类识别** — 每个槽位裁剪独立过 CLIP，输出 argmax 品类码 +
  余弦置信度（如 `s1 A · cos 0.31`），槽位卡片与视频 overlay 同步显示。
- **空位（缺货）检测** — 词表含 `EMPTY`（空货架 prompt）；槽位读到 EMPTY
  即判空，进入缺货状态机。余弦低于 `CLIP_MIN_COS`（默认 0.15）的帧**保持
  上一次观测**，一帧暗图/模糊不会误报缺货。
- **整屏网格计数（grid 模式，路径 A）** — `slot_mode: grid` 时一个大框
  覆盖整个取景区域，框内按 `rows × cols` 切成虚拟格子，逐格过同一个
  CLIP 引擎；帧内货物数 = 有货格子数（overlay `GOODS n/N` + 顶栏 Goods
  统计 + 右侧 "Goods in frame" 汇总卡，详见[计数模式](#计数模式槽位-vs-整屏网格)）。
- **逐件货物检测（chain A，yolo_world v5.4.0）** — 开放词表检测器在每个
  扫描 tick 对同一帧输出去重货物框（label + score + 所在格号），HD canvas
  与 Smooth MJPEG 同步渲染；新增 `ITEMS m` 计数与顶栏 Items 统计，
  `GOODS n/N` 语义不变。负样本命中（person/hand 等）画灰框、不计入。
- **排面歪斜（tilted）** — 槽位配置 `expected_code` 后，识别码 ≠ 计划码
  即标记歪斜（`plan A` 但读出 `E`），前端高亮。
- **销售情况热力图** — 时间桶 × 槽位 × 品类峰值计数聚合（sqlite 持久化），
  前端 1h / 6h / 1d / 7d 可切。
- **事件推送** — SSE：扫描快照 + `stockout` / `restock` 事件。
- **HD 预览（默认）** — platform-api 硬件 H.264 → 浏览器 MSE（~30 fps），
  槽位多边形与品类 chip 由前端 canvas 浮层绘制；抖动 / 无 token / 无 MSE
  自动降级 **Smooth**（服务器 MJPEG，overlay.py 服务端烘焙 chip）。

## 模型

| 项 | 值 |
|----|----|
| 模型 | `clip/clip_vit_b_32_image_encoder_nv12.hef`（CLIP ViT-B/32 图像塔） |
| 输入 | `input_layer1` **NV12** 224×224×3 uint8（flat 75264 字节） |
| 输出 | 512 × uint8 → 减均值 + 单位归一化后与文本嵌入比余弦 |
| 推理 | 设备实测 **≈ 48 ms/槽位**；10 槽位一轮 ≈ 0.5 s，应用跑 ~1–2 FPS |
| 文本侧 | 设备 `encode_text()` 现场生成（~2 s/prompt，一次），缓存 `.npy` |

注册必须 `model_type="embedding"` 且
`model_variant='{"backend_function": "identity"}'`（固件无 CLIP 后处理，
拿原始 uint8 输出自己解码）；注册在繁忙运行时上 ~1/3 概率失败，内置重试。
完整设备契约与证据链见 [docs/device-gate.md](docs/device-gate.md)。

文本嵌入矩阵为 `(N,512) float32 单位归一化`，缓存到
`EMBEDDING_PATH`（`.npy` + JSON sidecar `{codes, prompts, template}` 钉住
行序）；词表 / 模板 / 形状任一不匹配即重建，绝不串行序。离线可用
`tools/generate_embeddings.py`（open_clip 同构复现，x86 生成后
`EMBEDDING_MODE=file` 直接加载）。

### 逐件检测器（yolo_world v2s v5.4.0）

| 项 | 值 |
|----|----|
| 模型 | `yolo_world_v2s_540.hef`（开放词表检测，双输入） |
| 输入1 | `input_layer1` RGB 640×640×3 uint8（letterbox 居中黑边） |
| 输入2 | `input_layer2` **uint16** [1,80,512] 词表嵌入：`u16 = clip(rint(emb/2.7e-5 + 9778), 0, 65535)` |
| 输出 | NMS-by-class float32 流：per class `[count][count × 5 × (y1,x1,y2,x2,score)]` |
| 后处理 | 片上阈值 ≈0 → 主机阈值 `0.25` + 跨类 NMS IoU `0.45`（score 降序贪心） |
| 推理 | 设备实测 **≈ 0.24 s/帧**（每 tick 一次，与 CLIP 共用同一 4K 帧） |
| 词表 | 固定 80 行 = 40 货物 + 40 负样本（person/hand/shelf/…），随包 `assets/yolo_world_vocab.npy` + JSON sidecar（HF `openai/clip-vit-base-patch32` + clip_text.onnx 生成，probe50 验证） |

注册 `model_type="detection"` + identity variant + 显式 inputs，注册后
常驻不 unregister。失败语义：注册/加载失败**不致命**（`detector=None`，
CLIP 照跑，`/api/health` `detector:"unavailable"`）；单 tick 推理失败
保留上次框 ≤3 tick 后清空（`"degraded"`）。回滚 `DETECT_ENABLED=0`。
完整契约与平反证据链见 [docs/device-gate.md](docs/device-gate.md) §9。

## 架构

```
showcases/shelf-ops/
├── app.py               # Flask 入口：注册 CLIP、词表嵌入、后台扫描循环、SSE、HD/MJPEG 预览、/api/*
├── config.py            # ShelfConfig：env + config.yaml overlay；MODEL_DEFS + A–E prompt 词汇 + 默认 10 槽位
├── clip_classify.py     # NV12 转换 / 裁剪 / bbox padding / uint8 解码 / 注册重试 / SlotClassifier
├── detector.py          # yolo_world v5.4.0 逐件检测：letterbox / uint16 词表 / NMS 解析 / 格映射
├── vocab.py             # prompt 词汇表 + 文本嵌入矩阵生成/缓存/失配重建
├── slots.py             # 槽位多边形 / 网格虚拟格子 + per-slot 状态机（缺帧保持、EMPTY 码、tilted、goods_summary）
├── analytics.py         # 时间桶计数 + sqlite 持久化 + 热力图聚合
├── overlay.py           # OpenCV 帧绘制：槽位多边形 + 品类 chip / 网格大框 + GOODS·ITEMS / 逐件检测框（MJPEG 烘焙路径）
├── alerts.py            # 缺货/补货事件（SSE + 冷却）
├── app.yaml             # manifest：models 卷 + inference 权限 + events + network host:8891
├── Dockerfile           # python:3.11-slim + SDK wheel（无 torch —— 文本嵌入走设备 EncodeText）
├── config.example.yaml  # 10 槽位 demo 排面 + grid 计数 + 词汇 override 模板
├── assets/              # yolo_world 检测词表嵌入 npy + JSON sidecar（随包）
├── tools/generate_embeddings.py  # 离线词表嵌入生成（device / open_clip 两种模式）
├── docs/device-gate.md  # Stage 0 设备门禁 + 检测器证据链（§6–§9）
└── tests/               # pytest：clip_classify / detector / slots / analytics / vocab（65 passed）
```

平面部署双模式：`network.mode=host, inbound=8891` 直接 LAN 访问
`http://<host>:8891/`，亦经 app-manager `/apps/shelf-ops/` 反代 ——
前端全相对路径，两入口同一页面。

## 本地运行（模拟模式，无设备）

```bash
cd showcases/shelf-ops
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

SHELF_SIMULATE=1 \
CONFIG_PATH="$PWD/config.example.yaml" \
ANALYTICS_DB=/tmp/shelf-analytics.db \
WEB_PORT=8891 \
python3 app.py
```

模拟模式跳过设备注册，按固定 pattern 合成每槽位品类（A,B,C,D,E,A,C,E,
EMPTY,EMPTY 循环移位），并合成 3–5 个缓慢抖动的逐件检测框 —— overlay /
MJPEG / 热力图 / 事件全链路可用。

## 设备部署

```bash
# 1. 模型上机（镜像卷映射 /data/aipc/models，只读）
scp clip_vit_b_32_image_encoder_nv12.hef root@<host>:/data/aipc/models/clip/
scp yolo_world_v2s_540.hef root@<host>:/data/aipc/models/

# 2. 构建并安装 app bundle（参照仓库 scripts/build_showcase_artifacts.sh）
aipc-cli app install app.yaml shelf-ops-image.tar   # 升级用 app update 保住 /data/aipc/etc 卷

# 3. 槽位标定：把 config.example.yaml 拷成 /data/aipc/etc/shelf-ops/config.yaml
#    （多边形归一化坐标按现场货架取景调整），重启 app
```

### 运维注意（逐件检测）

- **相机对焦是前置条件**：失焦帧检出率崩塌（probe50 全部 <0.15，根因是
  光学失焦而非模型）。部署后 root ssh 上机执行
  `aipc-cli device focus auto`（Laplacian var 5.7 → 清晰度提升 10–60×）。
  取景状态栏 **"Autofocus: Off" 有误导** —— 那只是说没有连续 AF，
  一次性 AF 已生效，不用管它。
- **HEF 需在位**：`/data/aipc/models/yolo_world_v2s_540.hef` 缺失或注册
  失败 → 检测链关闭，app **降级纯 CLIP**（`GOODS` 照常，`/api/health`
  `detector:"unavailable"`），无需回滚镜像。
- **配置旋钮**（env 或 config.yaml）：`DETECT_ENABLED`（默认 1；**=0 整链
  关闭**，回到纯 CLIP 行为）、`DETECT_THRESHOLD`（0.25）、
  `DETECT_NMS_IOU`（0.45）、`DETECT_TIMEOUT_MS`（15000）。
- 词汇表随包发布（`assets/`），换检测词表需重新生成嵌入矩阵并同步
  sidecar（80 行 = 40 货物 + 40 负样本，行序钉死）。

## 计数模式：槽位 vs 整屏网格

`slot_mode` 选择计数方式（决策证据链见
[docs/device-gate.md](docs/device-gate.md) §8）：

| | `slots`（默认） | `grid`（路径 A） |
|--|--|--|
| 标定 | 逐槽位画多边形 + `expected_code` 计划码 | 一个大框 `region: [x1,y1,x2,y2]`（归一化） |
| 计数 | 每槽位 FULL / EMPTY / 歪斜检测 | 大框切 `rows × cols` 虚拟格子，货物数 = 有货格子数 |
| 适用 | 固定排面、要按位管理的货架 | 不想逐槽标定、只要"框里有多少货"的整屏 / 桌面场景 |

grid 模式的格子复用同一 CLIP 直分类引擎与状态机（每格 capacity 1，
id `g{行}-{列}`），`slots:` 配置整体被忽略。格子串行推理 ~48 ms/格：
3×4=12 格约 0.6 s/轮，`rows × cols` 上限 144。渲染上 overlay.py
（MJPEG 烘焙）与前端 canvas 是一对双胞胎 —— thick 大框 + 暗色空格
边线 + 有货格品类 chip + `GOODS n/N` 汇总；`/api/state` 增加 `mode`
与 `goods: {cells, occupied, by_code}`。

设备配置（`/data/aipc/etc/shelf-ops/config.yaml`）：

```yaml
slot_mode: grid
grid:
  region: [0.0, 0.0, 1.0, 1.0]   # 整屏；缩到桌面/货架区域即只数框内
  rows: 3
  cols: 4
  show_cells: false               # 整屏货架视图：隐藏格线/格 chip
```

env 覆盖：`SLOT_MODE` / `GRID_ROWS` / `GRID_COLS` / `GRID_SHOW_CELLS`。

`show_cells: false` 把整个 region 渲染成**一个货架**：只保留大框 +
`GOODS n/N · ITEMS m` 汇总 + 逐件检测框，格线与 `gR-C` 格 chip 全部隐藏
（仅渲染层——格子仍在内部驱动 CLIP 计数，`/api/state` 的 `slots` 数组与
热力图不受影响）。默认 `true` 显示格子。

grid 汇总条为 `GOODS n/N · ITEMS m`，两个计数语义不同、互相独立：
`GOODS n/N` 是格 CLIP 计数（有货格数 / 总格数），`ITEMS m` 是逐件检测的
去重货物框数（中心落在 region 内）。检测链关闭或不可用时 ITEMS 省略
（顶栏 Items 显示 `-`），GOODS 不受影响。

## 词汇与摆台

A–E 词表每类一个 CLIP prompt（`config.py` 内置默认，`config.yaml`
`vocabulary` 可覆盖）；`EMPTY` 是特殊空位码。分类是 prompt 级语义：

| 代码 | 显示名 | prompt 语义 | 摆台建议（demo） |
|------|--------|-------------|------------------|
| **A** | 瓶装水 | 透明塑料水瓶 | 塑料/玻璃水瓶 |
| **B** | 罐装饮品 | 铝制易拉罐 | 易拉罐汽水/啤酒 |
| **C** | 盒装零食 | 硬纸盒包装 | 盒装零食/盒装奶 |
| **D** | 瓶装饮品 | 塑料软饮瓶 | 高颈饮料瓶 |
| **E** | 水果 | 苹果橙子类鲜果 | 苹果/橙子（颜色区分最稳） |

槽位 capacity 语义为**二值**（每槽位每 tick 至多一个识别码）：
`capacity: 1` 的槽位读到商品即 FULL、读到 EMPTY 即空；capacity > 1 的
槽位有货时读 PARTIAL。换品类只改 prompt —— 首次启动自动重新生成嵌入缓存。

## 测试

```bash
cd showcases/shelf-ops
python3 -m pytest tests/ -q    # 65 passed
```

覆盖：NV12/裁剪/解码与注册重试（clip_classify）、yolo_world 量化/NMS
解析/letterbox 几何/格映射/端到端 detect 与注册重试（detector）、槽位
状态机（占→空→补、缺帧保持、tilted）、网格切格与 `goods_summary`
（slots）、analytics 时间桶聚合与 retention、
vocab 词汇映射与缓存失配重建。
