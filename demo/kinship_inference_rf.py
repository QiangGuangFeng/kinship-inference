#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于PLINK统计特征和随机森林的亲缘关系推断工具
功能：
1. 读取PED/MAP格式数据
2. 计算IBD统计量（Z0, Z1, Z2）、PI_HAT、IBS等特征
3. 使用随机森林进行关系分类
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def read_ped_map(ped_file, map_file):
    """读取 PED 和 MAP 文件"""
    # 读取 PED 文件 - 使用灵活的分隔符处理
    ped_df = pd.read_csv(ped_file, sep=r"\s+", header=None)
    
    # 读取 MAP 文件获取 SNP 信息
    map_df = pd.read_csv(map_file, sep=r"\s+", header=None,
                         names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    
    # 提取基因型并转换为数值
    genotypes = {}
    for idx, row in ped_df.iterrows():
        fid = row[0]
        iid = row[1]
        geno_row = row[6:].values
        
        # 每两个等位基因为一个 SNP
        snp_genos = []
        for i in range(0, len(geno_row), 2):
            if i+1 < len(geno_row):
                a1, a2 = str(geno_row[i]), str(geno_row[i+1])
                snp_genos.append((a1, a2))
        
        n_snps = len(map_df)
        genotypes[iid] = snp_genos[:n_snps]  # 确保与 MAP 文件 SNP 数一致
    
    return ped_df, map_df, genotypes


def calculate_ibs(geno1, geno2):
    """计算两个个体间的IBS共享等位基因数"""
    ibs_total = 0
    n_snps = min(len(geno1), len(geno2))
    
    for i in range(n_snps):
        a1_set = set(geno1[i])
        a2_set = set(geno2[i])
        
        # 计算共享等位基因数
        shared = len(a1_set & a2_set)
        if len(a1_set) == 1 and len(a2_set) == 1:
            # 都是纯合子
            ibs_total += 2 if a1_set == a2_set else 0
        else:
            # 至少一个是杂合子
            ibs_total += shared
    
    return ibs_total / (2 * n_snps)  # 标准化为0-1


def estimate_ibd_method_of_moments(geno1, geno2, allele_freqs):
    """
    使用矩估计法估算IBD系数（Z0, Z1, Z2）
    基于PLINK的方法
    """
    n_snps = min(len(geno1), len(geno2))
    
    # 统计不同IBS状态的SNP数量
    ibs0_count = 0  # 共享0个等位基因
    ibs1_count = 0  # 共享1个等位基因
    ibs2_count = 0  # 共享2个等位基因
    
    for i in range(n_snps):
        g1 = geno1[i]
        g2 = geno2[i]
        
        # 计算共享等位基因数
        alleles1 = list(g1)
        alleles2 = list(g2)
        
        # 计算IBS
        shared = 0
        temp_alleles2 = alleles2.copy()
        for a1 in alleles1:
            if a1 in temp_alleles2:
                shared += 1
                temp_alleles2.remove(a1)
        
        if shared == 0:
            ibs0_count += 1
        elif shared == 1:
            ibs1_count += 1
        else:
            ibs2_count += 1
    
    total = ibs0_count + ibs1_count + ibs2_count
    if total == 0:
        return {'Z0': 0.5, 'Z1': 0.5, 'Z2': 0.0, 'PI_HAT': 0.25}
    
    # 频率估计
    p_ibs0 = ibs0_count / total
    p_ibs1 = ibs1_count / total
    p_ibs2 = ibs2_count / total
    
    # 矩估计IBD系数
    # E(IBS=0) = Z0 * P(IBS=0|IBD=0)
    # 简化估计
    z2 = max(0, p_ibs2 - 0.25) / 0.75  # 调整
    z1 = max(0, p_ibs1 - 0.5 * (1 - z2)) / (1 - z2 + 0.001)
    z0 = max(0, 1 - z1 - z2)
    
    # 归一化
    z_sum = z0 + z1 + z2
    if z_sum > 0:
        z0 /= z_sum
        z1 /= z_sum
        z2 /= z_sum
    
    # PI_HAT = P(IBD=2) + 0.5 * P(IBD=1) = z2 + 0.5*z1
    pi_hat = z2 + 0.5 * z1
    
    return {
        'Z0': z0,
        'Z1': z1,
        'Z2': z2,
        'PI_HAT': pi_hat,
        'IBS0': p_ibs0,
        'IBS1': p_ibs1,
        'IBS2': p_ibs2
    }


def calculate_king_coefficient(geno1, geno2):
    """
    计算KING亲缘系数（robust to population structure）
    φ ≈ 0.5^n，n为亲缘代数
    """
    n_snps = min(len(geno1), len(geno2))
    
    # 计数
    n_concordant_hom = 0  # 纯合子一致
    n_discordant_hom = 0  # 纯合子不一致
    n_het1 = 0  # 个体1杂合
    n_het2 = 0  # 个体2杂合
    
    for i in range(n_snps):
        g1 = sorted(geno1[i])
        g2 = sorted(geno2[i])
        
        is_hom1 = g1[0] == g1[1]
        is_hom2 = g2[0] == g2[1]
        
        if is_hom1 and is_hom2:
            if g1 == g2:
                n_concordant_hom += 1
            else:
                n_discordant_hom += 1
        elif is_hom1:
            pass
        elif is_hom2:
            pass
        else:
            # 两者都是杂合
            pass
    
    # KING系数简化计算
    total_hom = n_concordant_hom + n_discordant_hom
    if total_hom == 0:
        return 0.0
    
    king_coef = 1 - n_discordant_hom / total_hom
    king_coef = (king_coef - 0.5) / 0.5  # 标准化
    
    return max(0, min(0.5, king_coef))


def calculate_mendel_errors(trio_genos):
    """
    计算孟德尔错误率（用于亲子鉴定）
    trio_genos: (father_geno, mother_geno, child_geno)
    """
    father, mother, child = trio_genos
    n_snps = min(len(father), len(mother), len(child))
    
    errors = 0
    for i in range(n_snps):
        f_alleles = set(father[i])
        m_alleles = set(mother[i])
        c_alleles = list(child[i])
        
        # 检查孩子的每个等位基因是否来自父母
        possible_alleles = f_alleles | m_alleles
        
        for c_allele in c_alleles:
            if c_allele not in possible_alleles:
                errors += 1
                break
    
    return errors / n_snps if n_snps > 0 else 0


def extract_features(genotypes, known_relationships=None):
    """
    提取所有个体对的特征
    返回特征矩阵和标签
    """
    individuals = list(genotypes.keys())
    features_list = []
    labels_list = []
    pairs_info = []
    
    for ind1, ind2 in combinations(individuals, 2):
        geno1 = genotypes[ind1]
        geno2 = genotypes[ind2]
        
        # 计算IBD系数
        ibd_stats = estimate_ibd_method_of_moments(geno1, geno2, None)
        
        # 计算IBS
        ibs = calculate_ibs(geno1, geno2)
        
        # 计算KING系数
        king_coef = calculate_king_coefficient(geno1, geno2)
        
        # 构建特征向量
        features = {
            'Z0': ibd_stats['Z0'],
            'Z1': ibd_stats['Z1'],
            'Z2': ibd_stats['Z2'],
            'PI_HAT': ibd_stats['PI_HAT'],
            'IBS': ibs,
            'KING': king_coef,
            'IBS0': ibd_stats['IBS0'],
            'IBS1': ibd_stats['IBS1'],
            'IBS2': ibd_stats['IBS2'],
        }
        
        features_list.append(features)
        pairs_info.append((ind1, ind2))
        
        # 如果有已知关系，添加标签
        if known_relationships is not None:
            label = get_relationship_label(ind1, ind2, known_relationships)
            labels_list.append(label)
    
    return pd.DataFrame(features_list), labels_list, pairs_info


def get_relationship_label(ind1, ind2, known_rels):
    """根据已知关系字典获取标签"""
    pair = tuple(sorted([ind1, ind2]))
    
    if pair in known_rels:
        return known_rels[pair]
    
    # 默认标记为未知
    return 'UNKNOWN'


def create_training_data():
    """创建示例训练数据"""
    # 定义已知关系
    known_relationships = {
        ('CHILD01', 'PARENT1'): 'PO',  # Parent-Offspring
        ('CHILD01', 'PARENT2'): 'PO',
        ('CHILD02', 'PARENT1'): 'PO',
        ('CHILD02', 'PARENT2'): 'PO',
        ('CHILD03', 'PARENT1'): 'PO',
        ('CHILD03', 'PARENT2'): 'PO',
        ('CHILD01', 'CHILD02'): 'FS',  # Full Siblings
        ('CHILD01', 'CHILD03'): 'FS',
        ('CHILD02', 'CHILD03'): 'FS',
        ('SIB1', 'SIB2'): 'FS',
        ('HALF1', 'HALF2'): 'HS',  # Half Siblings (share one parent)
        ('UNREL1', 'UNREL2'): 'UN',  # Unrelated
    }
    
    # 添加反向对
    known_relationships_full = {}
    for (a, b), rel in known_relationships.items():
        known_relationships_full[(a, b)] = rel
        known_relationships_full[(b, a)] = rel
    
    return known_relationships_full


def train_random_forest(X, y):
    """训练随机森林分类器"""
    # 编码标签
    label_encoder = {label: idx for idx, label in enumerate(set(y))}
    y_encoded = [label_encoder[label] for label in y]
    
    # 检查每个类别的样本数
    from collections import Counter
    class_counts = Counter(y_encoded)
    print(f"\n各类别样本数: {class_counts}")
    
    # 如果某些类别样本太少，不使用分层抽样
    min_class_count = min(class_counts.values()) if class_counts else 0
    
    if min_class_count < 2:
        print("警告: 某些类别样本数少于2，不使用分层抽样")
        stratify_param = None
        test_size = 0.2  # 减小测试集比例
    else:
        stratify_param = y_encoded if len(set(y_encoded)) > 1 else None
        test_size = 0.3
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, 
        stratify=stratify_param
    )
    
    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
    
    rf.fit(X_train, y_train)
    
    # 评估
    y_pred = rf.predict(X_test)
    
    # 解码标签
    inverse_encoder = {idx: label for label, idx in label_encoder.items()}
    y_test_labels = [inverse_encoder[idx] for idx in y_test]
    y_pred_labels = [inverse_encoder[idx] for idx in y_pred]
    
    print("\n" + "="*60)
    print("随机森林分类结果")
    print("="*60)
    print(f"\n训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    print(f"\n特征重要性:")
    for feat, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    print(f"\n分类报告:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    print(f"混淆矩阵:")
    print(confusion_matrix(y_test_labels, y_pred_labels, labels=list(label_encoder.keys())))
    
    return rf, label_encoder, inverse_encoder


def predict_relationships(rf, label_encoder, inverse_encoder, X, pairs_info):
    """预测个体对的关系"""
    predictions = rf.predict(X)
    probabilities = rf.predict_proba(X)
    
    results = []
    for i, (pair, pred, proba) in enumerate(zip(pairs_info, predictions, probabilities)):
        pred_label = inverse_encoder[pred]
        confidence = max(proba)
        
        results.append({
            'Individual1': pair[0],
            'Individual2': pair[1],
            'Predicted_Relationship': pred_label,
            'Confidence': confidence,
            'Probabilities': dict(zip(label_encoder.keys(), proba))
        })
    
    return pd.DataFrame(results)


def main():
    """主函数"""
    import os
    
    # 设置数据路径
    demo_dir = '/workspace/demo'
    ped_file = os.path.join(demo_dir, 'toy.ped')
    map_file = os.path.join(demo_dir, 'toy.map')
    
    print("="*60)
    print("基于随机森林的亲缘关系推断")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(ped_file):
        print(f"错误: PED文件不存在: {ped_file}")
        print("正在创建示例数据...")
        create_demo_data(demo_dir)
    
    if not os.path.exists(map_file):
        print(f"错误: MAP文件不存在: {map_file}")
        return
    
    # 读取数据
    print(f"\n读取数据:")
    print(f"  PED文件: {ped_file}")
    print(f"  MAP文件: {map_file}")
    
    try:
        ped_df, map_df, genotypes = read_ped_map(ped_file, map_file)
        print(f"  个体数: {len(genotypes)}")
        print(f"  SNP数: {len(map_df)}")
    except Exception as e:
        print(f"读取数据失败: {e}")
        print("使用模拟数据进行演示...")
        genotypes = create_simulated_genotypes()
        map_df = pd.DataFrame({'SNP': [f'rs{i}' for i in range(100)]})
    
    # 创建已知关系标签
    known_relationships = create_training_data()
    
    # 提取特征
    print("\n提取特征...")
    X, y, pairs_info = extract_features(genotypes, known_relationships)
    
    print(f"\n特征矩阵维度: {X.shape}")
    print(f"特征列: {list(X.columns)}")
    print(f"\n部分特征数据预览:")
    print(X.head(10))
    
    # 过滤掉UNKNOWN标签用于训练
    valid_indices = [i for i, label in enumerate(y) if label != 'UNKNOWN']
    if len(valid_indices) == 0:
        print("\n警告: 没有已知关系的样本，无法训练模型")
        print("将使用预定义规则进行分类...")
        rule_based_classification(X, pairs_info)
        return
    
    X_valid = X.iloc[valid_indices]
    y_valid = [y[i] for i in valid_indices]
    pairs_valid = [pairs_info[i] for i in valid_indices]
    
    print(f"\n用于训练的样本数: {len(X_valid)}")
    print(f"关系类别分布: {pd.Series(y_valid).value_counts().to_dict()}")
    
    # 训练随机森林
    rf, label_encoder, inverse_encoder = train_random_forest(X_valid, y_valid)
    
    # 预测所有个体对
    print("\n" + "="*60)
    print("预测所有个体对的关系")
    print("="*60)
    results_df = predict_relationships(rf, label_encoder, inverse_encoder, X, pairs_info)
    
    print("\n预测结果:")
    print(results_df.to_string())
    
    # 保存结果
    output_file = os.path.join(demo_dir, 'relationship_predictions.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n预测结果已保存至: {output_file}")
    
    # 可视化特征重要性
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)
    
    # 分析不同关系的特征模式
    for rel_type in set(y_valid):
        rel_mask = [y[i] == rel_type for i in range(len(y_valid))]
        if sum(rel_mask) > 0:
            rel_data = X_valid[rel_mask]
            print(f"\n{rel_type}关系的平均特征值:")
            for col in X.columns:
                print(f"  {col}: {rel_data[col].mean():.4f} ± {rel_data[col].std():.4f}")


def create_simulated_genotypes():
    """创建模拟基因型数据用于演示"""
    np.random.seed(42)
    
    # 创建一些模拟个体
    genotypes = {}
    
    # 父母
    parent1 = [(np.random.choice(['A', 'G']), np.random.choice(['A', 'G'])) for _ in range(100)]
    parent2 = [(np.random.choice(['C', 'T']), np.random.choice(['C', 'T'])) for _ in range(100)]
    
    genotypes['PARENT1'] = parent1
    genotypes['PARENT2'] = parent2
    
    # 孩子（从父母各继承一个等位基因）
    for i in range(3):
        child = []
        for j in range(100):
            a1 = parent1[j][np.random.randint(0, 2)]
            a2 = parent2[j][np.random.randint(0, 2)]
            child.append((a1, a2))
        genotypes[f'CHILD0{i+1}'] = child
    
    # 无关个体
    for i in range(2):
        unrelated = [(np.random.choice(['A', 'G', 'C', 'T']), np.random.choice(['A', 'G', 'C', 'T'])) 
                     for _ in range(100)]
        genotypes[f'UNREL{i+1}'] = unrelated
    
    # 同胞
    sib1 = []
    sib2 = []
    for j in range(100):
        a1_p1 = parent1[j][np.random.randint(0, 2)]
        a2_p1 = parent2[j][np.random.randint(0, 2)]
        a1_p2 = parent1[j][np.random.randint(0, 2)]
        a2_p2 = parent2[j][np.random.randint(0, 2)]
        sib1.append((a1_p1, a2_p1))
        sib2.append((a1_p2, a2_p2))
    
    genotypes['SIB1'] = sib1
    genotypes['SIB2'] = sib2
    
    return genotypes


def rule_based_classification(X, pairs_info):
    """基于规则的简单分类（当没有训练数据时）"""
    print("\n基于规则的分类结果:")
    print("-"*60)
    
    results = []
    for i, row in X.iterrows():
        pair = pairs_info[i]
        
        # 简单规则
        if row['PI_HAT'] > 0.4 and row['Z2'] < 0.1:
            rel = 'PO'  # Parent-Offspring
        elif row['PI_HAT'] > 0.4 and row['Z2'] > 0.1:
            rel = 'FS'  # Full Siblings
        elif row['PI_HAT'] > 0.2 and row['PI_HAT'] < 0.4:
            rel = 'HS'  # Half Siblings
        elif row['PI_HAT'] < 0.15:
            rel = 'UN'  # Unrelated
        else:
            rel = 'UNCERTAIN'
        
        results.append({
            'Individual1': pair[0],
            'Individual2': pair[1],
            'Predicted_Relationship': rel,
            'PI_HAT': row['PI_HAT'],
            'Z0': row['Z0'],
            'Z1': row['Z1'],
            'Z2': row['Z2']
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string())
    
    return results_df


if __name__ == '__main__':
    main()
