# World Model Pipeline

程序化飞机场景、CWM + MiniMax-H3 视频生成，以及 UE RGB 视频外观增强。

这是从已完成实验中整理出的独立源码目录。默认路径相对于本仓库，不依赖上一层目录的实验输出。项目代码位于 `src/world_model_pipeline/`，实际使用的第三方源码位于 `third_party/`，本地权重位于 `models/`。

## 目录

```text
assets/                 示例参考图片、模型转换所需的结构头
configs/                场景配置、模型文件与 SHA-256 清单
docs/                   流程、迁移和验证说明
models/                 实际复制的权重和 Qwen processor；Git 忽略
src/world_model_pipeline/
  geometry/             粗飞机、OBB、相机和深度/语义渲染
  workflows/            飞机实验与 UE 分段的配置准备
  inference/            CWM/H3、原生 RGB Ref2VA 和视频导出
  orchestration/        GPU 选择、单任务与批量任务监督
  diagnostics/          平面几何与 RGB 光流诊断
  tools/                直接下载、H3 共享组件转换
third_party/
  code_world_model/     CWM 发布代码和原有许可/说明
  musubi_tuner/         实际使用的 Musubi 源码，包含已应用补丁
tests/                  CPU 回归与可移植性测试
outputs/                运行结果；Git 忽略
data/                   外部输入数据；Git 忽略
reports/                本地校验记录；Git 忽略
```

## 环境

已验证环境为 Python 3.10、PyTorch 2.9.1 + CUDA 12.8。生成使用单张 A100 80 GB 和 PyTorch FLASH SDPA，默认交换 48 个 DiT block。批处理只使用显式允许且空闲的 GPU。

新机器创建环境时使用清华镜像：

```bash
WMP_PYTHON=python3.10 bash scripts/bootstrap_env.sh
source .venv/bin/activate
python scripts/wmp doctor
python scripts/wmp models verify
```

`requirements.lock.txt` 是当前已验证环境的精确版本快照。PyTorch、NumPy 等发行包通过该文件安装，不把整个虚拟环境复制进 Git。源码形式的 CWM/Musubi 使用本仓库 `third_party/`，优先于环境中其他安装。

已有兼容 Python 环境时，也可以直接 `python scripts/wmp --help`，不必先安装本项目。命令从其他工作目录执行时，使用 `scripts/wmp` 的完整路径即可。

## 飞机：粗模型与 OBB

```bash
python scripts/wmp proxy generate --scene configs/aircraft_scene.json --output outputs/aircraft_proxy
python scripts/wmp obb --source outputs/aircraft_proxy --output outputs/aircraft_obb
```

粗模型为 174 顶点、292 三角形。OBB 是机体坐标系包围盒，随飞机姿态变换，8 顶点、12 三角形；并非最小体积拟合。输出包含 RGB 预览、336×192 的相机 Z 深度、语义类别、飞机轨迹和相机参数。

先准备并运行单个视频：

```bash
python scripts/wmp aircraft prepare --proxy outputs/aircraft_proxy --anchor assets/examples/aircraft/anchor_photoreal.png --output outputs/aircraft_single
python scripts/wmp aircraft run --config outputs/aircraft_single/config.json --gpu 4
```

复现 8 种背景/天气 + 2 种 OBB 对照：

```bash
python scripts/wmp aircraft prepare-suite --proxy outputs/aircraft_proxy --anchor assets/examples/aircraft/anchor_photoreal.png --output outputs/aircraft_suite --gpus 4,5,6,7
python scripts/wmp aircraft run-suite --directory outputs/aircraft_suite
```

8 组实验固定晴天参考图；雨、雪和夜景文字可能与它冲突。09 使用 OBB + 写实飞机图，10 使用 OBB + 方块图。每条生成 124 帧、1344×768、24 fps，约 5.17 秒。默认场景会渲染 214 帧供更长序列使用；这一套 10 条实验使用前 124 帧。

底层 CWM 阶段也可单独运行：

```bash
python scripts/wmp cwm validate --config configs/cwm.example.json
python scripts/wmp cwm prepare --config configs/cwm.example.json --gpu 4
python scripts/wmp cwm generate --config configs/cwm.example.json --gpu 4 --crf 18
```

生成缓存与视频仍使用 `.a100_sdpa` 后缀，以明确区分实验性 SDPA 后端。新 CWM 导出显式使用 CRF 18，并额外保存三个压缩前 PNG。此导出设置改动不表示已经重新生成了历史实验。

## UE：Observer07 / Observer08

```bash
python scripts/wmp ue prepare --source /path/to/ue_capture --anchors-dir assets/examples/observers --output outputs/observer_suite --gpus 4,5,6,7
python scripts/wmp ue run-suite --directory outputs/observer_suite
```

输入结构与限制见 [UE 输入格式](docs/PIPELINES.md#ue-输入格式)。当前实现复现已验证的 30 fps、300 帧、10 秒 Observer 数据。数据源通过 `--source` 显式传入；仓库中没有写死原机器的数据目录。

每条原视频变成 240 帧、24 fps，分成两个重叠 8 帧的 124 帧片段。实际 RGB 参考是 672×384，生成结果是 1344×768。这个基线使用原生 H3 Ref2VA，不加载 CWM LoRA。示例目录中的四张参考图对应之前的 Observer07/08 数据；新场景需要自行提供对应参考图。

参考图名为 `<Observer名称>_seg01.png` / `_seg02.png`。若未提供，准备阶段仍会导出 `source_first.png`，生成前需要补齐各 job 的 `anchor_photoreal.png`。图像生成工具本身的服务模型权重不在本地，本仓库保存已生成的参考图，并接受自有图片输入。

## 串联视频

```bash
python scripts/wmp video concat --inputs outputs/observer_suite/Observer07_OrbitRiseSlow/enhanced_stitched.mp4 outputs/observer_suite/Observer08_OrbitFallSlow/enhanced_stitched.mp4 --output outputs/observer_07_08.mp4
```

输入应为同尺寸、同帧率、同编码的无声视频。使用视频流直接拼接，并验证全片帧数、时间戳及片段边界像素，保留原画质。

## 测试和 Git

```bash
python -m pytest -q
python scripts/check_repo.py
git status --short
git check-ignore models/diffusion_models/minimax_h3_ref2va_bf16.safetensors
```

权重已实际复制到 `models/`，不使用符号链接或硬链接；`configs/model_manifest.json` 记录每个文件的尺寸和 SHA-256。完整重验可用 `python scripts/wmp models verify --full`，会读完约 118 GiB 数据。

`.gitignore` 排除权重、视频、数据、虚拟环境、缓存和运行记录；代码、第三方源码、参考图片、配置及模型清单进入 Git。项目仓库：[LeeCASC/code_wm_datasets_pipeline](https://github.com/LeeCASC/code_wm_datasets_pipeline)。克隆仓库后需要另行准备本地模型文件。第三方许可和版本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 精度边界

相机与飞机 JSON 精确描述程序化 proxy 或原始 UE 帧。神经生成后的 RGB 尚不能自动继承这些 GT。已完成 UE 实验存在明显视角漂移与分段接缝；飞机新增背景物体也没有程序化三维轨迹。当前运动是脚本运动学，不是已校准的空气动力学仿真。

更多细节：[实际流程](docs/PIPELINES.md)、[迁移映射](docs/MIGRATION.md)、[本次验证](docs/VALIDATION.md)。
