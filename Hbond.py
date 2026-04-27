import os
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch

def get_sidechain_hbond_status(pdb_path, target_res_keys):
    """
    仅分析侧链原子形成的氢键状态
    0: 无, 1: 仅供体, 2: 仅受体, 3: 既是供体也是受体
    """
    parser = PDBParser(QUIET=True)
    if not os.path.exists(pdb_path):
        return {}
    
    structure = parser.get_structure('protein', pdb_path)
    
    # 1. 提取所有侧链原子
    # 排除主链原子: N, CA, C, O (以及 OXT)
    mainchain_atoms = {'N', 'CA', 'C', 'O', 'OXT'}
    all_atoms = list(structure.get_atoms())
    sidechain_atoms = [a for a in all_atoms if a.get_name() not in mainchain_atoms]
    
    # 2. 定义侧链极性原子（简易模型）
    # 侧链供体 (Donors): 包含 N 或 O 的侧链原子
    side_donors = {'NG', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OG', 'OG1', 'OH', 'SG'}
    # 侧链受体 (Acceptors): 包含 O 或 N 的侧链原子
    side_acceptors = {'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'ND1', 'NE2'}

    # 建立空间搜索索引（仅包含侧链原子，用于判断侧链间的相互作用）
    ns = NeighborSearch(sidechain_atoms)
    
    # 结果字典 {(chain, resnum): status_code}
    hbond_results = {key: 0 for key in target_res_keys}

    for atom in sidechain_atoms:
        res = atom.get_parent()
        chain_id = res.get_parent().get_id()
        res_id = res.get_id()[1] # 获取残基序号
        res_key = (chain_id, res_id)

        if res_key not in target_res_keys:
            continue

        # 搜索 3.5A 范围内的其他侧链原子
        neighbors = ns.search(atom.get_coord(), 3.5)
        
        is_donor = False
        is_acceptor = False

        for nb_atom in neighbors:
            nb_res = nb_atom.get_parent()
            # 排除同残基内部原子
            if nb_res == res:
                continue
            
            atom_name = atom.get_name()
            nb_name = nb_atom.get_name()

            # 判断当前残基侧链原子作为供体
            if atom_name in side_donors and nb_name in side_acceptors:
                is_donor = True
            # 判断当前残基侧链原子作为受体
            if atom_name in side_acceptors and nb_name in side_donors:
                is_acceptor = True

        # 更新状态码
        current_status = hbond_results[res_key]
        if is_donor and is_acceptor:
            hbond_results[res_key] = 3
        elif is_donor:
            hbond_results[res_key] = 1 if current_status != 2 else 3
        elif is_acceptor:
            hbond_results[res_key] = 2 if current_status != 1 else 3

    return hbond_results

def process_batch_hbond(csv_dir, pdb_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 匹配 acc_ 开头的 CSV 文件
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith('acc_') and f.endswith('.csv')]

    for csv_file in csv_files:
        # 解析文件名: acc_ID_Chain_Site.csv
        # 例如: acc_A0A1B2JLU2_A_3.csv -> ID 为 A0A1B2JLU2
        parts = csv_file.split('_')
        pdb_id = parts[1]
        
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"Skipping {csv_file}: PDB not found.")
            continue

        print(f"Processing Sidechain H-Bond: {csv_file}")
        
        df = pd.read_csv(os.path.join(csv_dir, csv_file))
        
        # 准备待查询的残基列表 (链, 序号)
        target_keys = set(zip(df['chain'].astype(str), df['resnum'].astype(int)))
        
        # 执行分析
        hbond_map = get_sidechain_hbond_status(pdb_path, target_keys)
        
        # 映射回 DataFrame
        df['hbond_type'] = df.apply(lambda x: hbond_map.get((str(x['chain']), int(x['resnum'])), 0), axis=1)
        
        # 保存文件
        out_name = csv_file.replace('acc_', 'hbond_sidechain_')
        df.to_csv(os.path.join(output_dir, out_name), index=False)


# ---------------- 配置 ----------------
if __name__ == "__main__":
    ACC_CSV_DIR = "./pos-results/sasa_data"  # 上一步生成的 acc_*.csv 文件夹
    PDB_FILES_DIR = "../pdbs"          # 存放 PDB 文件的文件夹
    HBOND_OUTPUT_DIR = "./pos-results/hbond_data" # 最终输出文件夹

    process_batch_hbond(ACC_CSV_DIR, PDB_FILES_DIR, HBOND_OUTPUT_DIR)
