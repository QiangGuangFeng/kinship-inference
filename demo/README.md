# 基于随机森林的亲缘关系推断工具

## 概述
本工具使用 scikit-learn 的随机森林算法，基于 PLINK 计算的统计特征（IBD 系数、PI_HAT、IBS 等）进行亲缘关系推断。

## 功能特点
1. **读取 PED/MAP 格式数据**：支持标准 PLINK 格式输入
2. **自动特征计算**：
   - IBD 系数 (Z0, Z1, Z2)
   - PI_HAT (亲缘系数)
   - IBS 统计量 (IBS0, IBS1, IBS2)
   - KING 亲缘系数
3. **随机森林分类**：自动训练模型并预测关系类型
4. **关系类型**：
   - PO (Parent-Offspring): 亲子关系
   - FS (Full Siblings): 全同胞
   - HS (Half Siblings): 半同胞
   - UN (Unrelated): 无关个体

## 使用方法

### 准备数据
```bash
# 确保有 PED 和 MAP 文件
# toy.ped - 基因型数据
# toy.map - SNP 位置信息
```

### 运行脚本
```bash
python kinship_inference_rf.py
```

### 输出结果
- 控制台输出：特征重要性、分类报告、混淆矩阵、各类别特征模式
- CSV 文件：`relationship_predictions.csv` 包含所有个体对的预测结果

## 依赖
```bash
pip install scikit-learn pandas numpy
```

## 示例数据说明
`toy.ped` 包含以下个体：
- PARENT1, PARENT2: 父母
- CHILD01, CHILD02, CHILD03: 子女（与父母构成亲子关系）
- SIB1, SIB2: 同胞对
- HALF1, HALF2: 半同胞
- UNREL1, UNREL2: 无关个体

## 判断标准参考
| 关系类型 | PI_HAT | Z0 | Z1 | Z2 |
|---------|--------|----|----|----|
| 亲子 (PO) | ~0.5 | ~0 | ~1.0 | ~0 |
| 全同胞 (FS) | ~0.5 | ~0.25 | ~0.5 | ~0.25 |
| 半同胞 (HS) | ~0.25 | ~0.5 | ~0.5 | ~0 |
| 无关 (UN) | ~0 | ~1.0 | ~0 | ~0 |

## 注意事项
1. 需要足够的已知关系样本用于训练（建议每类至少 5-10 个样本）
2. SNP 数量越多，推断越准确（建议>1000 个 SNP）
3. 小样本情况下可能影响模型性能，建议使用规则-based 方法作为补充
