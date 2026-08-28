# App 与模型管理改进方案（评审稿）

- 状态：**评审中，未实施**——本文档只含调查结论与建议方案，平台仓库
  （`ne503-aipc`）相关条目均未动代码
- 日期：2026-08-26
- 来源：build.sh 构建问题诊断延伸出的四个架构问题
- 仓库约定：`platform/...`、`web/...` 路径指平台仓库 `ne503-aipc`；
  其余路径指本仓库 `ne503-aipc-apps`

## 结论速览

| # | 问题 | 调查结论 | 建议落点 |
|---|------|---------|---------|
| 1 | 权限声明没人 get 到 | schema 齐全、**容器级真执行**（沙箱能力裁剪/dma_heap/host 网络），但**服务级 ACL 基本是 TODO**（只打日志）→ 声明了平台也不会拒绝什么 | apps：权限参考文档+模板注释；平台：3 个执行点接线 |
| 2 | app 能否用自己的模型 | **已经支持**（`allow_register_model` → ai-runtime OwnerId → DB OwnerAppID → 卸载自动清理）。缺的是声明式依赖：安装时不校验模型在位，运行时才失败 | 平台：`spec.models` **alias 映射**（id/path）+ 安装时校验 |
| 3 | 仓库依赖过大 | 唯一硬依赖是**构建期 SDK wheel**（干净克隆必挂）；模型、requirements.txt 均已解耦 | apps：SDK 从 PyPI 安装 `neoruntime-ipc-sdk`（已落地；早期过渡方案为 wheel 内置 `third_party/`，见附录 A，现已移除） |
| 4 | 导入方式 2/3 能否统一 | 判断正确：**两条路最终同落点（`apps/manifests/<id>/app.yaml`）、同安装器（AsyncInstallApp）**，方式 2 就是方式 3 的可视化生成器 | 平台：upload-manifest 返回全量解析 + 向导 hydrate + YAML 保真回写 |

---

## Q1：为什么用户「没 get 到」权限管理

### 现状（证据）

权限执行分两层，现状完全不同：

**容器级（真实执行）** — `platform/app-manager/security/sandbox.go`：
- 所有 app 默认 drop 11 项 capabilities（SYS_ADMIN/NET_ADMIN/MKNOD…）、
  完整 namespace 隔离、PidsLimit 128、CPU/内存配额
- `video` 权限 → 才挂 `/dev/dma_heap`（只读）+ 共享 IPC namespace
- `network.mode: host` → 才共享主机网络栈（默认 isolated）
- 声明即生效，这部分权限管理是真实的

**服务级（几乎未执行）** — 根因所在：
- `platform/app-manager/server/server.go` `registerAppPermissions()`：
  inference.models **只打一行日志**，源码注释
  `TODO: Implement actual permission registration API in AI Runtime`
- `sandbox.go` 对**所有 app 无条件挂载 `/run/aipc`**（camera/event/inference
  全部 IPC socket）→ 任何 app 都能直连三个守护进程
- event-bus 无 per-app topic ACL；camera-daemon 无 per-app 流白名单；
  ai-runtime 不校验 `inference.models`/`max_qps`/`max_concurrent`
- `platform/app-manager/manifest/manifest.go` 的
  `HasPermission()`/`CanPublishEvent()`/`CanSubscribeEvent()` 是死代码
  （仅测试调用）

### 根因

用户感受「权限声明是走形式」，是因为**声明与约束之间今天只连着容器级那一段**。
events.publish、inference.models 这些字段的实际效果是「装完没人管」——
不声明也能发事件、也能推理，声明了也不会被限流。

### 改进方案

**apps 仓库（可实施）**：
1. 新增 `docs/app-permissions.md`：六类权限（video/inference/events/device/
   network/security）逐项写清「声明语法 + 平台实际行为 + 为什么要声明」，
   明确标注哪些当前真实生效、哪些是安装审查清单/未来执行点
2. `templates/basic/app.yaml` 加注释块：每个权限段一句话 rationale

**平台仓库（3 个执行点）**：
1. event-bus：连接握手带 app 身份，publish/subscribe 按声明 topic 过滤
   （`/run/aipc` 挂载可收敛为按需挂载）
2. camera-daemon：订阅流时按 `permissions.video` 白名单校验
3. ai-runtime：`inference.models` 注册白名单 + `max_qps`/`max_concurrent`
   限流（完成 server.go 里的 TODO；`HasPermission` 等死代码接上或删除）

---

## Q2：app 自带模型与统一管理

### 现状（证据）

- **app 注册模型已支持**：`allow_register_model: true` → app 经 ai-runtime
  注册模型；ai-runtime 记 `OwnerId` → platform-api 同步到 DB
  `OwnerAppID`（`platform/platform-api/model/ai_model.go:37`）；卸载时
  `DeleteByOwnerAppID` 自动清理（`platform/platform-api/handlers/app.go:627`）；
  web 模型页已能显示归属
- 模型文件实际有两条共存路径：
  - **平台托管**：`/data/aipc/models`（web 上传/预置），app 经
    `inference.models` 声明引用（shelf-ops 即此路径，HEF 只读卷挂载）
  - **app 自注册**：文件通常来自挂载卷，生命周期随 app

### 缺口

没有**声明式依赖**：app.yaml 说不出「我需要 `clip_vit_b_32`，缺了起不来」。
安装时不校验 → 运行时才失败（shelf-ops 靠降级 + README 手工 scp 兜底）。

### 改进方案（结构已评审前沟通确认）

manifest 增加 `spec.models`：**alias → 映射**，key 即 app 代码里唯一写死的
名字，换模型只改 app.yaml、不改代码：

```yaml
spec:
  models:
    model1:                      # alias
      id: clip_vit_b_32          # 映射到平台已托管模型（web 上传/预置）
    model2:
      id: my_detector            # 注册时使用的模型 id
      path: /app/models/foo.hef  # 镜像内置模型文件
```

语义：
- **`id` 单独出现**：alias 解析为平台托管模型 id；安装时对照 DB /
  `/data/aipc/models` 校验在位
- **`id` + `path`**：app 自带模型，文件在镜像内；app 启动时由平台以该 id
  注册（复用现有 `allow_register_model` → OwnerId → `OwnerAppID` →
  卸载 `DeleteByOwnerAppID` 链路），alias 解析到它
- 与 `permissions.inference.models`（行为授权）互补互通：`spec.models`
  声明的 id 自动并入授权清单，接上 Q1 的执行点后即成白名单

**解析通道**（近期默认 env 注入，零 SDK 改动）：app-manager 创建容器时注入
`AIPC_MODEL_<alias>=<实际id>`，代码 `os.environ["AIPC_MODEL_model1"]` 取用
——即把 shelf-ops 今天 `CLIP_MODEL` env 的手工模式标准化为平台行为。
演进路径：per-app `models.json` 挂载（可携带输入输出规格元信息）→
SDK `resolve(alias)` API。

配套：
1. 安装时校验：`id` 查 DB；`path` 经镜像 tar 文件清单可检；缺失项在安装
   进度 UI 明确报出（必缺阻断 vs 可选降级 → 开放问题 2）
2. 展示层统一 provenance：模型页统一标注 `system / app-owned(by <id>)`

---

## Q3：apps 仓库解耦

依赖盘点结论：
- **构建期 SDK wheel**：唯一硬依赖（详见附录 A）
- **CI 私有 SDK 仓库 token**：wheel 内置后可一并去除
- **运行期模型**：设备卷挂载，本就解耦（镜像不打包模型是正确设计，保持）
- **requirements.txt**：不含 SDK（仅 Dockerfile wheel COPY 引入），无额外耦合

---

## Q4：导入方式 2/3 统一

### 现状（证据）

web 控制台 `web/src/pages/apps/components/ImportAppDialog.tsx` 三种来源，
后端两条路、一个落点：

- 方式 1/2（registry / 上传镜像）→ 6 步向导 →
  `POST /api/v1/apps/wizard` → `platform-api/handlers/wizard.go`
  `generateAppYAML()` 手拼 YAML → 存 `RootPath()/apps/manifests/<id>/app.yaml`
- 方式 3（上传 app.yaml + 可选 image.tar）→ **跳过全部编辑步骤** →
  `POST /api/v1/apps/upload-manifest` 原样存同一目录约定 →
  `POST /api/v1/apps/install-package`
- 两条路最终都调同一个 `AsyncInstallApp` gRPC

**判断成立**：方式 2 是方式 3 的可视化——它生成的正是方式 3 上传的东西。

### 改进方案

以 **app.yaml 为唯一事实源**，向导退化为「该 YAML 的可视化编辑器」：
1. `UploadManifest` 返回**完整解析后的 manifest**（复用 app-manager
   `manifest.LoadManifest` 作为唯一解析器，替代现在只解析 metadata 的
   匿名结构体）
2. 方式 3 上传后把解析值 hydrate 进向导各步骤，可审阅可微调，不改则直接
   安装——UI 收敛为「一种导入一个向导」，源选择变成第一步里的开关
3. **编辑回写必须保真**：在原 YAML 上做字段级 patch（Go yaml.v3 Node
   round-trip 保留注释与未知字段），**严禁**拿 WizardConfig 重建——它
   覆盖不了 healthcheck / restart_max_retries / security / multi-container /
   plugin 字段，重建即丢失
4. 顺手修 `generateAppYAML()` 的注入缺陷：现为 `strings.Builder` 手拼无
   转义，name 含 `:` 或引号直接产出非法 YAML → 改为构造
   `manifest.AppManifest` struct 后 `yaml.Marshal`（一处 schema 定义，
   前后端共享）

---

## 实施路线（建议分轮）

| 轮次 | 内容 | 仓库 |
|------|------|------|
| R1 | 附录 A：build.sh 修复 + wheel 内置（+可选 CI 切换） | apps |
| R2 | Q1 apps 侧：`docs/app-permissions.md` + 模板注释 | apps |
| R3 | Q4：manifest 全量返回 + 向导 hydrate + generateAppYAML 重写 | 平台 |
| R4 | Q2：`spec.models` alias 映射（id/path + env 注入）+ 安装校验 + provenance 展示 | 平台 |
| R5 | Q1 平台侧：三个服务级执行点接线（工作量最大，单独立项） | 平台 |

排期与分工由评审定。

## 开放问题（请评审时定）

1. Q1 服务级 ACL（R5）是否立项？不立项则 apps 侧文档需明确标注
   「events/inference 声明暂为审查清单，不构成运行时约束」
2. Q2 校验失败语义：必缺模型**阻断安装**还是警告放行（当前 shelf-ops 是
   设计为可降级的）
3. Q4 编辑回写取服务端 patch（保注释、实现重）还是前端 js-yaml
   round-trip（实现轻、丢注释）？
4. Q3 CI 是否同步切内置 wheel（涉及删 `SDK_REPO_TOKEN`，需 CI 负责人确认）

---

## 附录 A：build.sh 修复 + SDK wheel 内置（可直接实施）

**诊断**：`scripts/build_showcase_artifacts.sh` 的 `find_default_wheel()`
只搜 `dist/` 与兄弟 SDK 仓库（`../neoruntime-sdks/`、`../ne503-aipc-sdks/`），
`.gitignore` 又忽略 `dist/` 和 `*.whl` → **干净克隆跑 `./build.sh` 必报
"No SDK wheel found"**；本机能成功仅因 `dist/` 残留旧 wheel，CI 则靠
私有 SDK 仓库 token 现场构建。模型依赖是误解：镜像不打包模型，设备
`/data/aipc/models` 运行时只读挂载（93.72 上 CLIP / yolo_world HEF 均在位）。

**改动**：
1. 新增 `showcases/shelf-ops/build.sh`（照抄 gym-ops 11 行 wrapper，转发
   `scripts/build_showcase_artifacts.sh --arch arm64 --output dist/showcases shelf-ops`）
2. 新增 `showcases/shelf-ops/.dockerignore`（gym-ops 模板 + 排除 tests/、
   docs/、tools/、.venv/、config.yaml*、`*.npy`）
3. **wheel 内置**（历史方案，已被取代：`neoruntime-ipc-sdk` 已上架 PyPI，
   镜像改为构建期 `pip install neoruntime-ipc-sdk==<ver>`，`third_party/`
   与 `find_default_wheel()` 均已移除）：原做法为
   `third_party/neoruntime_ipc_sdk-<ver>.whl`（82KB 纯 Python 包，
   `git add -f`）+ 来源说明；`find_default_wheel()` 搜索序为
   `--wheel > third_party/ > dist/ > 兄弟仓库`。版本以已部署镜像实测
   `pip show hailo-ipc-sdk` 为准（候选 0.3.0 / 0.4.0）
4. （可选）CI 删 SDK checkout/构建步骤与 `SDK_REPO_TOKEN`
5. README：本地构建改为 `./build.sh` 一条命令；注明构建不需要模型

**验证**：`bash -n` 脚本 → 干净克隆 `./build.sh` 端到端（零依赖产出
`shelf-ops-0.3.0-arm64.tar.gz` + SHA256SUMS）→ 镜像内 SDK 版本抽查 →
设备 `aipc-cli app update`（保 `/data/aipc/etc` 配置卷）→
`curl :8891/api/health` 确认 `detector:"ok"`。

**明确不做**：HEF 打进镜像；设备上装 docker daemon（设备只有 containerd，
构建只在开发机/CI）。
