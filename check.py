import os
import pandas as pd

def CheckUnprocessedData(csv_path, output_dir):
    """
    检查CSV中的数据是否都生成了对应的结果文件
    :param csv_path: 输入的索引CSV文件路径 (nega_frg.csv)
    :param output_dir: 结果文件输出目录 (对应脚本中的 RESULT_FOLDER)
    :return: 未生成结果的条目列表
    """
    # 1. 读取索引CSV
    try:
        df_index = pd.read_csv(csv_path)
        id_col = df_index.columns[0]  # 第一列是PDB ID
        site_col = "Chain-Site"       # Site列名（和原脚本保持一致）
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return []

    # 2. 获取已生成的结果文件列表
    generated_files = []
    if os.path.exists(output_dir):
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".csv") and "_" in file_name:
                # 提取文件名中的 PDB ID 和 Site 信息（匹配原脚本的命名规则：{pdb_id}_{site_str.replace(':', '_')}.csv）
                generated_files.append(file_name)
    else:
        print(f"错误：输出目录 {output_dir} 不存在！")
        return []

    # 3. 逐行检查CSV中的条目是否有对应结果文件
    unprocessed = []
    for index, row in df_index.iterrows():
        pdb_id = str(row[id_col])
        site_str = str(row[site_col])
        
        # 生成预期的结果文件名（和原脚本命名规则一致）
        expected_filename = f"{pdb_id}_{site_str.replace(':', '_')}.csv"
        
        # 检查文件是否存在
        if expected_filename not in generated_files:
            unprocessed.append({
                "行号": index + 2,  # 行号从2开始（CSV表头占1行）
                "PDB ID": pdb_id,
                "Site": site_str,
                "预期文件名": expected_filename
            })

    # 4. 输出检查结果
    print("="*80)
    print(f"检查完成！总计CSV条目数: {len(df_index)}")
    print(f"已生成结果文件数: {len(generated_files)}")
    print(f"未成功生成的条目数: {len(unprocessed)}")
    print("="*80)

    if unprocessed:
        print("\n【未生成结果的详细信息】")
        # 转换为DataFrame便于格式化输出
        df_unprocessed = pd.DataFrame(unprocessed)
        print(df_unprocessed.to_string(index=False))
        
        # 可选：将未生成的条目保存为CSV
        unprocessed_csv = "未生成结果的条目.csv"
        df_unprocessed.to_csv(unprocessed_csv, index=False, encoding="utf-8")
        print(f"\n未生成条目已保存至: {unprocessed_csv}")
    else:
        print("\n✅ 所有CSV中的条目都已成功生成结果文件！")

    return unprocessed

# ------------------- 运行配置 -------------------
if __name__ == "__main__":
    # 请根据实际路径配置
    INPUT_CSV = "nega_frg.csv"          # 原索引CSV文件
    RESULT_FOLDER = "./neg-results"     # 原脚本的结果输出目录

    # 执行检查
    CheckUnprocessedData(INPUT_CSV, RESULT_FOLDER)