import os
import numpy as np
import pandas as pd
import iotbx.pdb
import mmtbx.model
from collections import defaultdict

# ------------------- 核心计算函数 -------------------

def GetAtoms(pdbf):
    """
    基础解析函数: 读取PDB并返回ATOM/HETATM行列表
    """
    with open(pdbf, 'r') as f:
        return [line for line in f.readlines() if line.startswith(("ATOM", "HETATM"))]

def GetSiteCenter(pdbf, site_str):
    """
    根据 site 字符串 (格式 '链:残基号', 如 'A:125') 获取残基中心坐标
    """
    try:
        target_chain, target_resnum = site_str.split(':')
        target_resnum = int(target_resnum)
    except ValueError:
        print(f"Warning: Site 格式错误 {site_str}，应为 'Chain:ResNum'")
        return None

    atomlines = GetAtoms(pdbf)
    coords = []
    for line in atomlines:
        chain = line[21]
        resnum = int(line[22: 26].strip())
        if chain == target_chain and resnum == target_resnum:
            x = float(line[30: 38].strip())
            y = float(line[38: 46].strip())
            z = float(line[46: 54].strip())
            coords.append([x, y, z])
    
    if not coords:
        print(f"Warning: 在 {pdbf} 中找不到 Site {site_str}")
        return None
    
    return np.mean(coords, axis=0) # 返回该残基的几何中心

def GetPocAtoms(pdbf, cx, cy, cz, radius=12.0):
    """
    提取中心点周围 radius 距离内的所有原子及其残基信息
    """
    atomlines = GetAtoms(pdbf)
    pocatoms = defaultdict(list)
    for line in atomlines:
        try:
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resnum = int(line[22:26].strip())
            ins_code = line[26].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            # 使用欧氏距离计算 (球形区域比立方体更精确)
            dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
            
            if dist <= radius:
                res_key = f"{chain}_{resname}_{resnum}{ins_code}"
                pocatoms["res_key"].append(res_key)
                pocatoms["chain"].append(chain)
                pocatoms["resname"].append(resname)
                pocatoms["resnum"].append(resnum)
                pocatoms["x"].append(x)
                pocatoms["y"].append(y)
                pocatoms["z"].append(z)
        except:
            continue
            
    return pd.DataFrame(pocatoms)

def FcalcAtAtoms(pdbf, atom_coords, resolution=2.0):
    """
    利用 Phenix 计算指定坐标处的电子密度
    """
    pdb_inp = iotbx.pdb.input(file_name=pdbf)
    model = mmtbx.model.manager(model_input=pdb_inp)
    xrs = model.get_xray_structure()
    
    fcalc = xrs.structure_factors(d_min=resolution).f_calc()
    fft_map = fcalc.fft_map(resolution_factor=0.25)
    fft_map.apply_volume_scaling()
    fcalc_map_data = fft_map.real_map_unpadded()
    uc = fft_map.crystal_symmetry().unit_cell()
    
    densities = []
    for p in atom_coords:
        frac = uc.fractionalize(p)
        densities.append(max(0, fcalc_map_data.value_at_closest_grid_point(frac)))
    return np.array(densities)

# ------------------- 批处理主程序 -------------------

def ProcessBatch(csv_path, pdb_dir, output_dir, temp_dir):
    # 准备环境
    for d in [output_dir, temp_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # 读取索引表
    df_index = pd.read_csv(csv_path)
    # 假设第一列是 ID，第二列是 Site
    id_col = df_index.columns[0]
    site_col = "Chain-Site"

    for index, row in df_index.iterrows():
        pdb_id = str(row[id_col])
        site_str = str(row[site_col])
        
        print(f"\n>>> 正在处理: {pdb_id} (Site: {site_str})")
        
        # 1. 查找PDB文件 (支持 .pdb 或 .ent)
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"Error: 找不到文件 {pdb_path}, 跳过...")
            continue

        # 2. 定位 Site 中心
        center = GetSiteCenter(pdb_path, site_str)
        if center is None: continue
        cx, cy, cz = center

        # 3. 为 Phenix 准备标准化的临时 PDB (加上晶胞信息)
        temp_pdb = os.path.join(temp_dir, f"tmp_{pdb_id}.pdb")
        with open(temp_pdb, "w") as f_out:
            f_out.write("CRYST1  150.000  150.000  150.000  90.00  90.00  90.00 P 1\n")
            with open(pdb_path, "r") as f_in:
                for line in f_in:
                    if line.startswith(("ATOM", "HETATM")):
                        # 设置 Occ=1, B=0
                        newline = line[:56]+"1.00"+line[60:61]+" 0.00"+line[66:]
                        f_out.write(newline)

        # 4. 提取周围残基原子
        poc_df = GetPocAtoms(pdb_path, cx, cy, cz, radius=12.0)
        if poc_df.empty: continue

        # 5. 计算原子密度
        coords = poc_df[["x", "y", "z"]].to_numpy()
        densities = FcalcAtAtoms(temp_pdb, coords)
        poc_df["density"] = densities

        # 6. 按残基聚合
        res_result = poc_df.groupby(["chain", "resname", "resnum", "res_key"]).agg({
            "density": "sum"
        }).reset_index()

        # 7. 保存结果
        output_name = f"{pdb_id}_{site_str.replace(':', '_')}.csv"
        res_result.to_csv(os.path.join(output_dir, output_name), index=False)
        print(f"Done! 结果已保存至: {output_name}")

        # 清理临时PDB
        if os.path.exists(temp_pdb): os.remove(temp_pdb)

# ------------------- 运行配置 -------------------

if __name__ == "__main__":
    # 配置你的路径
    INPUT_CSV = "nega_frg_2.csv"      # 你的索引表
    PDB_FOLDER = "../pdbs"       # 存放所有PDB文件的文件夹
    RESULT_FOLDER = "./neg-results-2" # 结果输出目录
    TEMP_FOLDER = "./temp_work" # 临时文件目录

    ProcessBatch(INPUT_CSV, PDB_FOLDER, RESULT_FOLDER, TEMP_FOLDER)
