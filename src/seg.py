import os, sys

def seg(args):
    input_file = os.path.join(args.datadir, "images")
    save_folder = os.path.join(args.datadir, "segmentation")
    os.makedirs(save_folder, exist_ok=True)
    config_fpath = "mseg-semantic/mseg_semantic/config/test/default_config_360_ms.yaml"
    os.system("python -u mseg-semantic/mseg_semantic/tool/universal_demo.py --config=%s --file_save 2 model_name mseg-3m model_path mseg-3m.pth input_file %s save_folder %s" %
              (config_fpath, input_file, save_folder))
