import os
import pandas as pd
import re

def parse_dssp(dssp_path):
    """
    解析 DSSP 文件，提取残基的可及表面积 (ACC)
    DSSP 格式说明:
    - 链 ID 在第 11 位 (index 11)
    - 残基序号在第 6-10 位 (index 5-10)
    - ACC (Solvent Accessibility) 在第 35-38 位 (index 34-38)
    """
    dssp_data = []
    with open(dssp_path, 'r') as f:
        lines = f.readlines()
        
    # 寻找数据开始的标志行 "#  RESIDUE AA STRUCTURE..."
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            start_idx = i + 1
            break
            
    for line in lines[start_idx:]:
        if len(line) < 40: continue # 跳过短行
        
        try:
            # DSSP 的列定义非常严格
            res_num_raw = line[5:10].strip()
            # 过滤掉非数字的残基号（如某些插入码情况，需根据实际DSSP版本微调）
            res_num = int(re.sub(r'\D', '', res_num_raw)) 
            chain = line[11].strip()
            acc = float(line[34:38].strip())
            
            dssp_data.append({
                'chain': chain,
                'resnum': res_num,
                'ACC': acc
            })
        except (ValueError, IndexError):
            continue
            
    return pd.DataFrame(dssp_data)

def merge_density_with_acc(csv_dir, dssp_dir, output_dir):
    """
    将计算好的密度 CSV 与 DSSP 数据合并
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有密度 CSV 文件
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    for csv_name in csv_files:
        # 从文件名解析 ID (假设格式为 ID_Chain_Site.csv)
        # 例如 2OO0_A_125.csv -> ID 为 2OO0
        pdb_id = csv_name.split('_')[0]
        
        # 寻找对应的 DSSP 文件 (支持 .dssp 或 .txt 后缀)
        dssp_path = os.path.join(dssp_dir, f"{pdb_id}.dssp")
        if not os.path.exists(dssp_path):
            # 尝试 .txt 后缀
            dssp_path = os.path.join(dssp_dir, f"{pdb_id}.txt")
            
        if not os.path.exists(dssp_path):
            print(f"Warning: 找不到 {pdb_id} 对应的 DSSP 文件，跳过...")
            continue

        print(f"正在处理: {csv_name} (匹配 DSSP: {pdb_id})")

        # 1. 读取密度 CSV
        density_df = pd.read_csv(os.path.join(csv_dir, csv_name))
        
        # 2. 解析 DSSP
        dssp_df = parse_dssp(dssp_path)
        
        if dssp_df.empty:
            print(f"Error: {dssp_path} 解析失败或为空")
            continue

        # 3. 合并数据
        # 注意：这里的 'chain' 和 'resnum' 必须与之前 CSV 中的字段名完全一致
        # 如果之前生成的 CSV 中 resnum 是字符串，这里需要统一类型
        density_df['resnum'] = density_df['resnum'].astype(int)
        dssp_df['resnum'] = dssp_df['resnum'].astype(int)
        
        merged_df = pd.merge(
            density_df, 
            dssp_df, 
            on=['chain', 'resnum'], 
            how='left'
        )

        # 4. 保存新 CSV
        output_path = os.path.join(output_dir, f"acc_{csv_name}")
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully saved to: acc_{csv_name}")

# ------------------- 运行配置 -------------------

if __name__ == "__main__":
    DENSITY_CSV_DIR = "./neg-results"      # 上一步生成的密度 CSV 文件夹
    DSSP_FILES_DIR = "../dssp"     # 存放 .dssp 文件的文件夹
    FINAL_OUTPUT_DIR = "./neg-results/sasa_data"  # 最终合并后的输出文件夹

    merge_density_with_acc(DENSITY_CSV_DIR, DSSP_FILES_DIR, FINAL_OUTPUT_DIR)
