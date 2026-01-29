#!/bin/bash

# 批量测试脚本：遍历指定目录下的所有 .pth 模型文件，并运行测试
# 用法：./Test.sh [--pth_dir PTH_DIR] [--base_dir BASE_DIR] [--log_path LOG_PATH]

# 默认参数
PTH_DIR="./Result/TransUNet_M/Chaos/Pth"
BASE_DIR="./Datasets/Chaos"
LOG_PATH="./Result/TransUNet_M/Chaos/Test.log"

# 检查参数是否存在
if [[ ! -d "$PTH_DIR" ]]; then
    echo "错误: PTH_DIR 不存在: $PTH_DIR"
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "错误: BASE_DIR 不存在: $BASE_DIR"
    exit 1
fi

# 确保日志文件所在目录存在
LOG_DIR=$(dirname "$LOG_PATH")
if [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    echo "创建日志目录: $LOG_DIR"
fi

echo "开始批量测试..."
echo "PTH_DIR: $PTH_DIR"
echo "BASE_DIR: $BASE_DIR"
echo "LOG_PATH: $LOG_PATH"
echo ""

# 查找所有 .pth 文件
PTH_FILES=("$PTH_DIR"/*.pth)
if [[ ${#PTH_FILES[@]} -eq 0 ]] || [[ ! -f "${PTH_FILES[0]}" ]]; then
    echo "错误: 在 $PTH_DIR 中未找到 .pth 文件"
    exit 1
fi

echo "找到 ${#PTH_FILES[@]} 个 .pth 文件:"
for f in "${PTH_FILES[@]}"; do
    echo "  $(basename "$f")"
done
echo ""

# 遍历每个 .pth 文件并运行测试
for pth_file in "${PTH_FILES[@]}"; do
    echo "========================================"
    echo "测试模型: $(basename "$pth_file")"
    echo "========================================"

    # 运行测试命令
    python Test.py --base_dir "$BASE_DIR" --model_path "$pth_file" --log_path "$LOG_PATH"

    # 检查退出状态
    if [[ $? -eq 0 ]]; then
        echo "测试完成: $(basename "$pth_file")"
    else
        echo "错误: 测试失败: $(basename "$pth_file")"
        # 可以选择继续测试下一个文件
    fi
    echo ""
done

echo "批量测试完成！所有结果已保存至: $LOG_PATH"