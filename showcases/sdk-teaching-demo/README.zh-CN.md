# SDK 教学演示 —— 一个可安装的交互课堂

单个 `.neoapp`，用"动手做"教 `neoruntime-ipc-sdk`：装上、打开页面，
对着真实摄像头流操作五个站点。不用先读文档——每个站点都标明它调用
的 SDK 接口，它防止的坑都是我们在真机上踩过的。

## 安装

```bash
# 设备控制台：应用 → 导入 → sdk-teaching-demo-latest-arm64.neoapp
# 或用 aipc-cli：
aipc-cli app install sdk-teaching-demo <app.yaml路径> <image.tar路径>
aipc-cli app start sdk-teaching-demo
```

浏览器打开 `http://<设备IP>:8090/`，或控制台 **Visit App**（经
`/apps/sdk-teaching-demo/` 反代，页面在反代下同样可用）。

模型说明：app 运行时注册内置的 `hailo_yolov8n_384_640.hef`（4 类：
person / vehicle / face / license_plate）。若设备已注册过
`yolov8n_384_640`，会直接复用现有注册。

## 五个站点

| # | 站点 | 可操作项 | 底层 SDK 调用 |
|---|------|---------|--------------|
| 1 | 实时检测（B 形态） | 引擎开关、本地画框、置信度阈值、类别过滤 | `FdMediaClient.subscribe("third", keep_fd=True)` → `InferencePipeline.run()` → `draw_detections()` → MJPEG |
| 2 | 渲染路径 | 本地绘制 vs 平台 overlay、教学区多边形 | `OverlayClient.annotate_result("main", result, ttl_ms)` vs 自己画像素 |
| 3 | 硬件路由 | 三种策略切换、拒绝演示 | `set_route_policy()`、`get_default_router().health()`、逐算子 hw/sw/fallback 增量 |
| 4 | A 形态 | 启停平台侧流水线 | `StreamPipeline("third", model, fps=10)` —— 平台订阅、推理并绘制 |
| 5 | 事件 | 观察带冷却门的再发布 | `EventClient.publish("app/sdk-teaching-demo/detection", …)`，5 秒冷却 |

页面默认英文，右上角可切换中文。

## 构建

```bash
./build.sh arm64   # → ../../dist/showcases/sdk-teaching-demo-<版本>-arm64.neoapp
```

`python:3.11-slim` + `neoruntime-ipc-sdk`（`sdk.lock` 锁定）+
`opencv-python-headless` + `numpy`。离线测试：`python3 -m pytest
tests/ -q`（无需 SDK 或设备——SDK 模块已打桩）。

## 权限（app.yaml）逐条解释

- `video: [main.raw, third.raw, sub.raw]` —— 订阅 `third` 做推理；
  在控制台 `main` 流上 annotate（归一化坐标跨同 FOV 流有效）。
- `inference.allow_register_model: true` —— 和 SDK 示例一样运行时
  `register_model()`。不配 `models:` 预载列表：运行时注册本身就是
  教学点之一。
- `events.publish: [inference/main, app/sdk-teaching-demo/*]` ——
  overlay 路径走 `inference/main`；app 自己的检测事件走自己的主题。
- 宿主 `/data/aipc/models` 只读同路径挂载 —— 注册路径对 ai-runtime
  必须是宿主真实路径。

## FAQ —— 内建进本 app 的契约

**引擎死亡为何退出码 1？** `restart_policy: on-failure` 把退出码 0
视为正常停止。无信号死亡的引擎返回 1，守护进程才会拉活 app，而不是
留一个僵尸页面。

**推理为什么用 `third` 不用 `main`？** 平台侧喂帧没有缩放器：流几何
必须等于运行时注册模型的输入。`third`（640×384）匹配；`main`
1080p 每帧 `DMA input rejected: -2811`（站点 4 有实测数字）。要喂
`main` 需平台管理模型（`.bin` 打包），不是 SDK 问题。

**为什么每次 annotate 都带 `ttl_ms`？** overlay 层会过期；app 按自己
节奏续发，停止后一秒内框自动消失。清理时用**空列表**——
`detections=None` 会保留残框。

**app.yaml 为什么没有 `models:` 列表？** 预载列表会变成启动期硬依赖，
还掩盖了本 app 要教的运行时 `register_model()`。镜像把 HEF 打到
`/opt/aipc/bundled-models` 作后备；宿主 `/data/aipc/models/detection/`
存在时优先。永不回退平铺根 `/data/aipc/models/`——某些设备上同名
文件是另一个编译版本。
