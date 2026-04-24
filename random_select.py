import os
import random
import shutil

# ========== 配置区 ==========
src_folder = r"F:\All_dataset\\long_buliao_datasets_260107_cutbackground"  # ← 请修改为你的源文件夹路径
dst_folder = r"F:\All_dataset\\260309buliao_datasets\\good"  # ← 请修改为目标文件夹路径
num_files = 328  # ← 要随机选取的文件数量
# ===========================


def random_select_files(src, dst, num):
    # 获取源文件夹中所有文件（仅顶层，不包括子文件夹）
    all_files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]

    if not all_files:
        print("❌ 源文件夹中没有文件。")
        return

    # 如果请求的数量大于实际文件数，自动调整
    actual_num = min(num, len(all_files))

    # 随机选取文件
    selected = random.sample(all_files, actual_num)

    # 创建目标文件夹（如果不存在）
    os.makedirs(dst, exist_ok=True)

    # 复制文件
    for filename in selected:
        src_path = os.path.join(src, filename)
        dst_path = os.path.join(dst, filename)
        shutil.copy2(src_path, dst_path)  # copy2 保留元数据
        print(f"✅ 已复制: {filename}")

    print(f"\n🎉 共成功复制 {len(selected)} 个文件到:\n{dst}")


if __name__ == "__main__":
    random_select_files(src_folder, dst_folder, num_files)
