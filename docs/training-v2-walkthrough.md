# 第二版训练：实际做了什么，以及你以后怎么操作

日期：2026-11-02。第二版数据整理、30 轮 GPU 训练、保存模型和新旧模型对比均已完成。

## 先分清四件事

1. **标注软件**负责画框、标类别，导出图片对应的 YOLO `.txt`。
2. **数据整理脚本**负责检查图片和标注、划分训练与验证，生成 `data.yaml`。
3. **训练脚本**调用 Ultralytics YOLO，让预训练模型学习这些标注，生成新的 `.pt` 权重。
4. **预测程序**加载训练好的 `.pt`，对图片或摄像头画面输出类别、框和置信度。

Docker 是训练环境。本次数据和代码保存在 Windows D 盘，通过挂载供 Docker 读取。没有在容器里额外保留一套需要手工取回的训练结果。

## 第一步：检查并整理第二版数据

使用的脚本：

`D:\invc\work\invc-yolo-garbage\scripts\prepare_self_dataset_v2.py`

在 Windows Python 中运行的命令：

```powershell
cd D:\invc\work\invc-yolo-garbage
python scripts/prepare_self_dataset_v2.py
```

**这一步已经完成。脚本会拒绝覆盖现有第二版数据，所以无需再次运行。** 该脚本按本次两个批次和已确认的留出场景编写，后续第三批照片应创建新的版本和划分记录。

它读取了：

- 第一版整理好的数据：`data/self_dataset/`，共 294 张。
- 新图片：`D:\invc\work\数据集11.2`，47 张。
- 新 YOLO 标注：`D:\invc\work\labels2`，47 份。
- 新图片旁的 JSON，用于交叉检查类别名称、框坐标和图片尺寸。
- 六类名称与顺序：`D:\invc\work\dataset\name.txt`。

图片与标注全部配对，类别顺序和坐标一致。没有改动原始图片、TXT、JSON，也没有改动第一版数据和权重。

输出目录：

```text
D:\invc\work\invc-yolo-garbage\data\self_dataset_v2\
├── images\train\          272 张训练图
├── images\val\             69 张验证图
├── labels\train\          对应训练标签
├── labels\val\            对应验证标签
├── data.yaml              给训练程序看的图片位置与类别配置
├── split_summary.json     各类数量、划分方式
└── source_manifest.json   来源路径及文件校验值
```

保留了第一版的 235 张训练图与 59 张验证图。新照片中的 37 张加入训练，另外 10 张按相近摆放场景成组留作验证，没有重新随机打乱旧照片。

| 类别 | 第二版训练图 | 第二版验证图 |
|---|---:|---:|
| 瓷片 | 71 | 19 |
| 石子 | 33 | 8 |
| 砖块 | 68 | 15 |
| 纸杯 | 68 | 24 |
| 土豆 | 32 | 6 |
| 塑料瓶 | 48 | 11 |

单张照片可以出现多类，所以此表各类数量之和不等于照片总数。

## 第二步：在 Docker 中训练

实际运行的训练脚本：

`D:\invc\work\invc-yolo-garbage\scripts\train_self.py`

在 Windows PowerShell 的项目目录运行：

```powershell
cd D:\invc\work\invc-yolo-garbage
docker compose run --rm --no-deps -T train python scripts/train_self.py --data /workspace/data/self_dataset_v2/data.yaml --epochs 30 --batch 8 --imgsz 640 --workers 4 --name v2_30ep
```

**上述训练已成功完成，不需要为了查看结果再次执行。** 以后重新做实验时，改用新的 `--name`，例如 `v2_30ep_repeat`，方便区分结果。

命令中各部分的意思：

| 部分 | 作用 |
|---|---|
| `docker compose run ... train` | 在现有训练镜像中运行程序 |
| `python scripts/train_self.py` | 执行训练入口 |
| `--data .../data.yaml` | 指定使用第二版数据 |
| `--epochs 30` | 最多训练 30 轮；本次完成全部 30 轮 |
| `--batch 8` | 每批处理 8 张训练图 |
| `--imgsz 640` | 使用 640 尺寸的训练输入 |
| `--workers 4` | 4 个数据加载进程 |
| `--name v2_30ep` | 本次实验结果文件夹名称 |

训练脚本内部主要执行：

```python
model = YOLO('/workspace/models/yolo11n.pt')
model.train(data='...', epochs=30, batch=8, imgsz=640, device='0', ...)
```

本次从官方 YOLO11n 预训练权重开始微调，使用 RTX 5060 Ti、PyTorch 2.8.0+cu128、CUDA 12.8、Ultralytics 8.4.164。优化器自动选择 AdamW，初始学习率 0.001，AMP 混合精度开启。

数据增强包括颜色与亮度变化、随机平移和缩放、左右翻转，以及低概率模糊等。前 20 轮使用 Mosaic，最后 10 轮关闭 Mosaic，其余增强继续使用。`args.yaml` 中的 `augment: false` 对应预测时增强，不是训练增强总开关。

## 第三步：找到生成的结果

本次结果全部保存在：

`D:\invc\work\invc-yolo-garbage\outputs\self_dataset\v2_30ep`

```text
v2_30ep\
├── weights\best.pt              验证指标选出的最佳模型，优先用于识别
├── weights\last.pt              最后一轮模型
├── results.png                  训练损失与验证指标曲线
├── results.csv                  每轮的数值
├── args.yaml                    本次训练参数
├── confusion_matrix.png         错分类、多余框、漏检统计
├── confusion_matrix_normalized.png
├── train_batch*.jpg             训练输入和标注预览
├── val_batch*_labels.jpg        验证图片上的真实标注
└── val_batch*_pred.jpg          验证图片上的预测框
```

完整终端日志另存于：`outputs/debug_logs/v2_30ep.log`。

Docker 内 `/workspace` 就是 Windows 的 `D:\invc\work\invc-yolo-garbage`。因此容器显示 `/workspace/outputs/...` 时，去 D 盘项目的 `outputs` 下查找即可。

第一版结果仍在 `outputs/self_dataset/debug_20261102_01/`。本次没有覆盖它。

## 第四步：确认补拍后的效果

运行脚本：`scripts/compare_saved_models_v2.py`。

该脚本对新旧 `best.pt` 使用同一批 69 张验证图，并额外比较相同预测参数下的框统计。对比结果保存在 `outputs/self_dataset/comparison_v1_v2/`。

| 同一验证集上的指标 | 第一版 | 第二版 |
|---|---:|---:|
| mAP@0.5 | 97.97% | 99.21% |
| mAP@0.5:0.95 | 90.45% | 93.94% |

以下框统计使用同一组参数：`conf=0.5, iou=0.5, agnostic_nms=True, imgsz=640, rect=False`。标注与预测匹配 IoU 固定为 0.5，全部 349 个真实目标。

| 统计 | 第一版 | 第二版 |
|---|---:|---:|
| 正确识别 | 324 | 335 |
| 错分类 | 7 | 1 |
| 多余未匹配框 | 6 | 6 |
| 未匹配真目标 | 18 | 13 |

新增留出的 10 张照片共 47 个目标，第一版正确识别 46 个，第二版 47 个。这只是这 10 张照片上的结果。

第二版同时增加了训练数据、训练轮数和 Mosaic，不能把改善全部归因于补拍。第一版保留下来的随机划分仍有相似场景跨集合的可能，新照片的背景和实物也较集中；这些数字尚不代表 USB 摄像头或新环境的准确率。

## 第五步：用第二版模型做识别

第二版图片预测配置已单独保存为 `configs/predict_self_v2.yaml`，其中 `model` 指向新的 `v2_30ep/weights/best.pt`，`source` 默认是第二版验证图片。

如需生成这批图片的识别效果图，在项目目录执行：

```powershell
docker compose run --rm --no-deps -T train yolo detect predict cfg=configs/predict_self_v2.yaml
```

这条命令做预测，不重新训练。首次结果保存在 `outputs/self_dataset/v2_30ep/predictions/`。本次已通过对比脚本加载新模型执行了预测，上面这条单独批量保存图片的命令尚未运行。

下一阶段是连接 USB 摄像头，使用第二版模型检查真实画面，再逐步接入垃圾大类映射、跟踪、测速和通信。
