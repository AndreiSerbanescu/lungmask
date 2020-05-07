import os
from common.utils import *
import shutil
from common_jip.batch_job import *
from listen import *


def handle_output(param_dict, element_output_dir):
    rel_segmentation_path = param_dict["segmentation"]
    data_share = os.environ["DATA_SHARE_PATH"]
    full_segmentation_path = os.path.join(data_share, rel_segmentation_path)

    element_output_name = os.path.join(element_output_dir, "lungmask.nii.gz")
    shutil.copyfile(full_segmentation_path, element_output_name)


if __name__ == "__main__":
    file_validator = NiftiAndDicomFileValidator(print_statements=True)
    start_batch_job(handle_output, run_lungmask_absolute, file_validator=file_validator)

