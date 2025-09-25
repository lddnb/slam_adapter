#!/bin/bash
# 允许脚本在失败时停止
set -e

# 使用宿主网络构建镜像
docker build --network=host --build-arg PROXY=http://127.0.0.1:7890  -t ubuntu-noble-dev:latest -f Dockerfile .