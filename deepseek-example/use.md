# 基于 deepseek 的案例

## 环境安装

> 前置条件: 安装好 Anaconda

- 创建虚拟环境

``` bash
# 创建一个 3.10 的 python 环境
conda create -n deepseek python=3.10
# 激活该环境
conda activate deepseek
# 移除环境
# conda remove --name deepseek --all --force
```

## 安装依赖

``` bash
# rich 包方便查看打印结果
pip install openai langchain langchain_community langchain-deepseek ollama rich
# 向量化需要的包
pip install torch  sentence-transformers chromadb
```

