import os
import glob
from listen import run_lungmask_absolute
from common.utils import *
import shutil

setup_logging()

batch_folders = [f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))]

for batch_element_dir in batch_folders:

    element_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR'])
    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    os.makedirs(element_output_dir, exist_ok=True)

    files = os.listdir(element_input_dir)

    if len(files) == 0:
        log_error(f"No files found in {element_input_dir} - skipping")
        continue

    if files[0].endswith(".nii.gz"):
        abs_source_file = os.path.join(element_input_dir, files[0])
    elif files[0].endswith(".dcm"):
        abs_source_file = element_input_dir
    else:
        log_error(f"Unrecognised input file type inside {element_input_dir} - skipping")
        continue

    try:
        param_dict, success = run_lungmask_absolute(abs_source_file)
    except Exception as e:
        log_error(f"Segmentation failed for {element_input_dir} with exception {e}")
        continue

    if not success:
        log_error(f"Segmentation failed for {element_input_dir}")
        continue

    rel_segmentation_path = param_dict["segmentation"]
    data_share = os.environ["DATA_SHARE_PATH"]
    full_segmentation_path = os.path.join(data_share, rel_segmentation_path)

    element_output_name = os.path.join(element_output_dir, "lungmask.nii.gz")
    shutil.copyfile(full_segmentation_path, element_output_name)
