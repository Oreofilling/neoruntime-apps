# 应用权限指南（app.yaml spec.permissions）

本文回答一个问题：**在 app.yaml 里声明权限，平台到底做了什么？**

平台仓库代码引用格式为 `文件:行号`（相对 platform 仓库根目录，如
`platform/app-manager/security/sandbox.go:191`）。所有行号基于
`feat/app-models-import-unify` 分支（R2+R3+R4 落地时）。

## 快速结论表

| 权限 | 现状 | 证据 |
|------|------|------|
| `resources.cpu` / `resources.memory` | **真实生效**（cgroup 配额/上限） | sandbox.go:145-169 |
| 11 项 capability drop + PidsLimit=128 + 私有 namespace | **真实生效**（对所有 app 无条件） | sandbox.go:106-127, :172 |
| `security.no_new_privileges` / `security.readonly_rootfs` | **真实生效但默认关闭** —— 必须显式写 `true`（见下文陷阱） | sandbox.go:129-143; runtime.go:352-357 |
| `permissions.video`（≥1 条流） | **部分生效** —— 触发 `/dev/dma_heap` 只读挂载；具体流不过滤 | sandbox.go:204-216 |
| `permissions.network.mode: host` | **真实生效** —— 共享主机网络 namespace | sandbox.go:218-221 |
| `permissions.network.outbound` | **未生效** —— 仅解析，无任何执行点 | manifest.go:130（全仓库唯一引用） |
| `permissions.events.publish/subscribe` | **未生效** —— event-bus 无 per-app ACL，只有 socket 组权限 | event-bus/server/main.go:480-484 |
| `permissions.inference.*` | **未生效**（运行时）—— 注册函数只打日志；`models` 经 R4A 并入授权清单并在安装期校验存在性 | server.go:251-276（:271 TODO） |
| `permissions.device.*`（light/ir_cut/ptz/lens/gpio） | **未生效** —— 设备控制走 `/run/aipc` IPC，声明与否不影响 | sandbox.go:191-202（无条件挂载） |
| `spec.models`（R4A 模型依赖） | **真实生效** —— 安装期校验 + 容器 env 注入 | server.go:2310+; runtime.go:187, :313 |

一句话总结：**容器隔离（namespace/caps/cgroup/挂载）是真的；服务级 ACL
（event/inference/device/outbound）尚未实现**（后者为 R5 立项范围）。

---

## 通用容器安全（无需声明，始终生效）

**平台实际行为**（单容器 sandbox.go:106-221；多容器 :257-421 同构，仅主容器获得 IPC）：

- 私有 PID / NET / IPC / UTS / Mount namespace（sandbox.go:106-112）
- 无条件 drop 11 项 capability：SYS_ADMIN、NET_ADMIN、SYS_MODULE、SYS_TIME、
  SYS_BOOT、SYS_NICE、SYS_RESOURCE、SYS_RAWIO、SYS_PTRACE、SYS_CHROOT、MKNOD
  （sandbox.go:114-127）
- PidsLimit = 128（sandbox.go:172）
- `/run/aipc`（camera / event / inference 全部 IPC socket 所在目录）对每个 app
  **无条件** bind 挂载、可写（sandbox.go:191-202）。这是"声明与否不影响设备/推理
  权限"的根因：socket 就在那里，服务端不做来源过滤。

## security 段

### 声明语法

```yaml
spec:
  security:
    no_new_privileges: true   # 阻止容器内提权（setuid 等）
    readonly_rootfs: true     # 根文件系统只读
```

### 平台实际行为

两项都被真实消费：runtime.go:356-357（`oci.WithNoNewPrivileges`）、
runtime.go:352-353（`oci.WithRootFSReadonly`）。

**陷阱：默认值是 `false`，不是注释里写的 `true`。**
`sandbox.go:129-131` 把两者初始化为 `false`，仅在 manifest 显式给出时覆盖
（sandbox.go:137-143）。`manifest.go:63-64` 的注释 "nil = true (default)" 与
实际行为不符 —— **省略 security 段 = 可写 rootfs + 允许提权**。

### 为什么要声明

这是六类权限中**唯一完全由 sandbox 层真实执行、但需要你主动打开**的开关。
生产应用建议始终带上上面两行 `true`。rootfs 只读时，可写路径只有 `spec.volumes`
声明的挂载点。

---

## video（视频流访问）

### 声明语法

```yaml
spec:
  permissions:
    video:
      - cam0_main.raw     # 原始流（dma-buf 零拷贝）
      - cam0_main         # 编码流（Unix socket）
```

### 平台实际行为

声明了至少一条流 → `/dev/dma_heap` 以**只读**方式挂载进容器
（sandbox.go:204-216，原始帧经 dma-buf fd 传递必需）。编码流走 `/run/aipc`
下的 Unix socket。

**未生效部分**：具体哪条流可访问不做过滤 —— `cam0_main.raw` 与
`cam1_sub.raw` 对已拿到 socket 的 app 无差别。

### 为什么要声明

现在就决定了你的镜像能否零拷贝取帧；将来的流级 ACL（R5）将以这份清单为准
回填执行，届时无需改代码。

---

## inference（推理权限）

### 声明语法

```yaml
spec:
  permissions:
    inference:
      models: [clip_vit_b_32, person_vehicle_v1]   # 授权模型 id 清单
      max_qps: 30
      max_concurrent: 2
      allow_register_model: false
```

### 平台实际行为

- `registerAppPermissions`（server.go:251-276）目前**只打日志**；
  server.go:271 的 TODO 原文即为此而留。qps / concurrent / allow_register_model
  无执行点。
- `models` 清单有两处真实作用：
  1. **R4A 之后**，`spec.models`（见下节）声明的依赖 id 会在解析时并入此清单
     （manifest.go `ParseManifest`，幂等、只改内存），Web 端应用详情页与安装
     向导展示的是并入后的结果。
  2. Web 端删除模型时会检查此清单（`used_by_apps`），被引用的模型禁止删除。

### 为什么要声明

模型清单已是**展示与防误删的真实数据**；安装期存在性校验（R4A）也读它。
qps/并发限制的 enforcement 在 R5。

---

## events（事件总线权限）

### 声明语法

```yaml
spec:
  permissions:
    events:
      publish: ["app/my_app/*", "alerts/*"]
      subscribe: ["model/*/detections", "system/*"]
```

### 平台实际行为

**未生效。** event-bus 服务端没有任何 per-app 鉴权 —— 唯一的"权限"是 socket
文件自身的组权限（event-bus/server/main.go:480-484），对所有 app 一致。
任何 app 都能对任意 topic 收发。

### 为什么要声明

作为**审查清单**：平台侧落地 topic 过滤（R5）时直接按此执行；对读者而言，
这份清单如实描述了 app 的事件面（审计/上架评审依据）。

---

## device（设备控制权限）

### 声明语法

```yaml
spec:
  permissions:
    device:
      light: true      # 白光灯
      ir_cut: true     # 红外滤片
      ptz: false       # 云台
      lens: false      # 变焦/聚焦
      gpio:
        read: [12, 13]
        write: [21, 22]
```

### 平台实际行为

**未生效。** 设备控制请求经 `/run/aipc` 下的 IPC socket 完成，而该目录对所有
app 无条件挂载（sandbox.go:191-202）；`device.*` 字段在 sandbox/runtime 中
零引用。

### 为什么要声明

同 events：声明是给 R5 执行点预留的契约，也是评审时回答"这个 app 能不能动
云台/白光"的权威答案。

---

## network（网络权限）

### 声明语法

```yaml
spec:
  permissions:
    network:
      mode: isolated              # isolated（默认）| host
      outbound: ["https://api.example.com"]
      inbound: [8554]             # host 模式下的入站端口说明
```

### 平台实际行为

- `mode: host` → **真实生效**：`NETNamespace = false`，容器直接用主机网络栈
  （sandbox.go:218-221）。`inbound` 仅作为文档性说明，无端口过滤。
- `outbound` → **未生效**：全仓库唯一出现处是结构体定义（manifest.go:130），
  无任何防火墙/代理执行点。isolated 模式 = 私有 netns + 默认容器网络，出站
  不受限。

### 为什么要声明

`mode` 决定真实的网络拓扑（RTSP 等入站服务必须 host）。outbound 清单是
最小权限意图的声明，R5 若加出站代理则以它为准。

---

## spec.models（声明式模型依赖，R4A 新增）

**这是取代"手配模型 id 环境变量"的正式机制。**

### 声明语法

```yaml
spec:
  models:
    clip:                    # alias：应用侧自选的名字（env 变量名后缀）
      id: clip_vit_b_32      # 平台模型 id（必填）
      path: /opt/aipc/bundled-models/clip_vit_b_32_image_encoder_nv12.hef
                             # 镜像内置模型文件（可选，Phase B 已生效）
      type: embedding        # 有 path 时必填（detection/embedding/depth/...）
      required: true         # 缺失时阻断安装（默认 false：仅告警）
```

约束（manifest.go `Validate()`）：alias 须匹配 `^[A-Za-z_][A-Za-z0-9_]*$`，
且不得为 `HOST_PREFIX` / `APP_ID` / `APP_ROLE` / `CONTAINER_NAME`（与平台保留
env 前缀冲突）；声明 `path` 时 `type` 必填且匹配 `^[a-z][a-z0-9_]*$`。

### 平台实际行为（四项均真实生效）

1. **安装期校验**（server.go:2310-2317 `validateModelDependencies`，调用点
   server.go:453 / :746）：一次 `ListModels` 全量比对 —— `required: true` 且
   设备缺失、manifest 又未给 `path` → 安装在**镜像拉取前**失败（不浪费几百
   MB 下载），所有缺失项合并成一条错误；`required: false` 且缺失 → 安装继续，
   任务进度中出现告警。
2. **env 注入**（runtime.go:313 单容器；:187 多容器全部容器）：容器创建时
   注入 `AIPC_MODEL_<alias>=<id>`。
3. **授权并入**：各依赖 id 并入 `permissions.inference.models`（解析时、仅
   内存），Web 展示与模型删除保护随之生效。
4. **镜像内置（`path`，Phase B 已生效）**：设备已有该 id → 平台侧副本优先，
   镜像文件不动作；设备缺失 → 安装时从镜像提取该文件（containerd
   ExtractFileFromImage）并以 **transient** 方式注册（模型页不显示、不写
   模型库）。

### path 的安全边界（决定哪些模型能进 spec.models）

transient 注册不携带 variant：检测类模型会被套默认 yolov8 后端（tensor 名
不匹配 → 推理期抛错），而应用通常把「已注册」视为良性、不再补注册 —— 没有
自愈路径。因此只有两类模型可以声明 `path`：

1. 原始输出无需后处理的模型（embedding / depth）—— 空 variant 即正确配置；
2. 应用启动时会带 variant 重新注册的模型（如 parking-lot 的
   yolov5m_vehicles：运行时重注册整体重写 postprocess 配置）。

需要复杂 postprocess 的模型必须留在应用侧运行时注册、不进 spec.models：
gym-ops 的 yolov8s_pose（native_yolov8_pose blob 只能经注册时的 variant
通道下发）、shelf-ops 的 yolo_world_540（identity variant + 双输入声明 +
应用侧 decode）、parking-lot 的车牌两件套（无 variant 时应用跳过重注册）。
这些模型靠 `permissions.inference.models` 授权清单 + 应用运行时注册；镜像
内文件仅作 `MODEL_ROOT` 缺失时的回退副本（host-first 解析）。

### 应用侧读取方式

```python
import os
model_id = os.environ["AIPC_MODEL_clip"]   # 不需要硬编码 clip_vit_b_32
```

迁移示例（shelf-ops）：app.yaml 删掉手配的 `CLIP_MODEL` 环境变量，改用上面的
声明 + 代码读 `AIPC_MODEL_clip`。平台升级/换模型时应用镜像零改动。

---

## 附：资源限制（无需 permissions，直接生效）

`spec.resources.cpu`（如 `"50%"`）→ CPU 配额；`spec.resources.memory`
（如 `"256Mi"`）→ 内存上限（sandbox.go:145-169，cgroup 强制）。

## 附：多容器应用

多容器模式（`spec.containers`）下安全配置同构（sandbox.go:249+），差异只有
一点：**IPC socket（/run/aipc）只挂给 `role: main` 的主容器**，子容器天然
拿不到平台服务 —— 子容器连 `permissions` 都不允许声明。
