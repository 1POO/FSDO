import os
import sys
import subprocess

def main():
    # 硬编码固定参数配置
    ckpt_path = ("/home/zfw/APP/lppp/TR-DETR-master/new_results/result_with_new_CTCloss4+0.3VTC_分开编码修正/hllinear/hl-video_sub_simple_tef-exp-2026_01_21_23_08_44/model_best.ckpt")
    eval_split_name = "test"
    # eval_split_name = "test"
    s_feat_dim = 512
    s_feat_dir = "/home/zfw/APP/lppp/TR-DETR-master/features/qvhighlights/clip_sub_features"
    eval_path = f"/home/zfw/APP/lppp/TR-DETR-master/data/highlight_{eval_split_name}_release.jsonl"
    a_feat_dir = "/home/zfw/APP/lppp/TR-DETR-master/features/qvhighlights/audio_features"  # 请确认实际路径
    a_feat_dim = 256  # 请确认实际维度

    # 调试信息打印
    print("[DEBUG] ckpt_path:", ckpt_path)
    print("[DEBUG] eval_split_name:", eval_split_name)
    print("[DEBUG] s_feat_dim:", s_feat_dim)
    print("[DEBUG] s_feat_dir:", s_feat_dir)
    print("[DEBUG] eval_path:", eval_path)
    print("[DEBUG] a_feat_dir:", a_feat_dir)
    print("[DEBUG] a_feat_dim:", a_feat_dim)

    # 环境变量配置
    project_path = "/home/zfw/APP/lppp/TR-DETR-master"
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{project_path}:{original_pythonpath}:."

    # 构建执行命令
    base_command = [
        "python",
        f"{project_path}/tr_detr/inference.py",
        "--resume", ckpt_path,
        "--eval_split_name", eval_split_name,
        "--eval_path", eval_path,
        "--s_feat_dim", str(s_feat_dim),
        "--s_feat_dir", s_feat_dir,
        "--a_feat_dir", a_feat_dir,
        "--a_feat_dim", str(a_feat_dim)
    ]

    # 处理额外参数
    if len(sys.argv) > 1:
        base_command.extend(sys.argv[1:])

    # 执行命令
    try:
        subprocess.run(base_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 执行失败，错误码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("[INFO] 用户中断执行")
        sys.exit(130)

if __name__ == "__main__":
    main()