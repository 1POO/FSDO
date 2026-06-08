import subprocess
import os

# 配置参数
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dset_name = "tvsum"
ctx_mode = "video_sub_simple_tef"
v_feat_types = "slowfast_clip"
t_feat_type = "clip"
s_feat_type = "clip"
results_root = "/home/zfw/APP/lppp/TR-DETR-master/results-tvsum/results1/test_set/FM_6666_2_2"
exp_id = "exp"
ctx_mode = "video_sub_simple_tef"
# 数据路径
train_path = "/home/zfw/APP/lppp/TR-DETR-master/data/tvsum/tvsum_train.jsonl"
eval_path = "/home/zfw/APP/lppp/TR-DETR-master/data/tvsum/tvsum_val.jsonl"
eval_split_name = "test"

# 特征路径
feat_root = "/home/zfw/APP/lppp/TR-DETR-master/features/tvsum"
v_feat_dim = 2048
t_feat_dim = 512
v_feat_dirs = [f"{feat_root}/video_features"]
v_objectFeat_dim = 2560
v_feat_dirs+=[f"{feat_root}/clip_object_features"]
t_feat_dir = f"{feat_root}/query_features/"

# 字幕特征
if s_feat_type == "clip":
    s_feat_dir = f"{feat_root}/sub_features"
    s_feat_dim = 512
else:
    raise ValueError("Wrong arg for s_feat_type.")

# 训练参数
bsz = 4
lr = 0.0005


# 域列表
# dset_domains = ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]
dset_domains = ["FM"]

# 可选：设置随机种子（可以手动设定具体值）
seed = 42
# 设置 PYTHONPATH
os.environ["PYTHONPATH"] = f"/home/zfw/APP/lppp/TR-DETR-master:{os.environ.get('PYTHONPATH', '')}"
# 开始训练
# 开始训练
for dset_domain in dset_domains:
    if dset_domain in [ "DS"]:
        VTC_loss_coef = 0.2
        CTC_loss_coef = 0.3
        seed = 42
    elif dset_domain in ["GA"]:
        VTC_loss_coef = 2
        CTC_loss_coef = 1.5
        seed = 42
    elif dset_domain in ["BT"]:
        VTC_loss_coef = 0.5
        CTC_loss_coef = 2
        seed = 8888
    elif dset_domain in ["PK"]:
        VTC_loss_coef = 1.5
        CTC_loss_coef = 1.5
        seed = 42
    elif dset_domain in ["MS"]:
        VTC_loss_coef = 0.3
        CTC_loss_coef = 2
        seed = 6666
    elif dset_domain in ["PR"]:
        VTC_loss_coef = 1.5
        CTC_loss_coef = 3.5
        seed = 42
    elif dset_domain in ["FM"]:
        VTC_loss_coef = 2
        CTC_loss_coef = 2
        seed = 6666
    elif dset_domain in ["BK"]:
        VTC_loss_coef = 3
        CTC_loss_coef = 2
        seed = 6666
    elif dset_domain in ["VU"]:
        VTC_loss_coef = 2
        CTC_loss_coef = 2.5
        seed = 8888
    elif dset_domain in ["VT"]:
        VTC_loss_coef = 0.5
        CTC_loss_coef = 1.1
        seed = 123
    else:
        VTC_loss_coef = 0.5
        CTC_loss_coef = 2
        seed = 42

    cmd = [
        "python", "/home/zfw/APP/lppp/TR-DETR-master/fsdo/train.py",
        "--VTC_loss_coef", str(VTC_loss_coef),
        "--CTC_loss_coef", str(CTC_loss_coef),
        "--dset_name", dset_name,
        "--ctx_mode", ctx_mode,
        "--train_path", train_path,
        "--eval_path", eval_path,
        "--eval_split_name", eval_split_name,
        "--v_feat_dirs", *v_feat_dirs,
        "--v_feat_dim", str(v_feat_dim),
        "--t_feat_dir", t_feat_dir,
        "--t_feat_dim", str(t_feat_dim),
        "--s_feat_dir", s_feat_dir,
        "--s_feat_dim", str(s_feat_dim),
        "--bsz", str(bsz),
        "--results_root", f"{results_root}_{dset_domain}",
        "--exp_id", exp_id,
        "--max_v_l", "1000",
        "--n_epoch", "2000",
        "--lr_drop", "2000",
        "--max_es_cnt", "-1",
        "--seed", str(seed),
        "--lr", str(lr),
        "--dset_domain", dset_domain
    ]


    # 调用训练脚本
    print(f"Running training for domain: {dset_domain}")
    subprocess.run(cmd)
