# 视频标注工具

在视频中进行目标检测的标注。框选第一帧目标roi区域后，使用图像跟踪器自动标注后续视频序列。如果跟踪失败或者跟踪偏差过大，则需要人工修正后继续跟踪。

---

编译运行：

软件第三方依赖库

| 第三方库 | 作用         | 安装方式                                         |
| -------- | ------------ | ------------------------------------------------ |
| OpenCV 4 | 图像处理库   | [github链接](https://github.com/opencv/opencv)   |
| libfmt   | 格式化字符串 | sudo apt install libfmt-dev                      |
| yaml-cpp | YAML文件读写 | [github链接](https://github.com/jbeder/yaml-cpp) |

使用cmake进行编译

```shell
mkdir build
cd build
cmake ..
make
```

运行

```shell
./LabelVideo <path to video> <path to yaml>
```

---

按键表

| 按键 | 功能                                   |
| ---- | -------------------------------------- |
| q    | 上一帧                                 |
| e    | 下一帧                                 |
| a    | 添加一个目标roi                        |
| d    | 删除选中的目标roi（鼠标单击roi框选中） |
| z    | 退出程序                               |

当按下a按键后，会弹出一个窗口用于选择roi，选择完毕后，需要输入该目标的类别名（只能包含大小写英文字符）和目标的ID（不同帧中，相同ID的roi表示同一个目标）。输入时，窗口右侧有对应提示信息。

---

图像ROI框颜色和对应含义

| 颜色 | 含义                       |
| ---- | -------------------------- |
| 绿色 | 该目标为之前标注好的目标   |
| 蓝色 | 该目标为追踪器追踪到的目标 |
| 红色 | 该目标被鼠标选中           |

