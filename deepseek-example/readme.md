
# 提示词

# 知识库

# FunctionCall

# FineTuning

# 微调

基于 LLamMA-Factor 微调

https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md

安装
``` bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
#启动,启动后会打开一个web页面
python src/webui.py

```

## 介绍

可以查看目录下有个 LLAMA-FACTOR 目录，目录中有几个重要的文件

- data 数据集文件夹

- savs

- src/webui.py

这个里面是启动界面，可以修改端口等信息

- 下载模型

``` bash
# 安装魔塔sdk
# https://modelscope.cn/models/deepseekai/DeepSeek-R1-Distill-Qwen-1.5B/files
pip install modelscope
```
然后可以运行 download.ipynb 下载模型



# 角色扮演

``` bash
# 安装 streamlit 框架(UI)
pip install streamlit

# 启动
streamlit run app.py
```
