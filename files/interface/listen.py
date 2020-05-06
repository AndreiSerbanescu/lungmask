import os
from lungmask import lungmask
from lungmask import utils
import SimpleITK as sitk
import numpy as np
import time
from common.utils import *
from common import listener_server



def run_lungmask(param_dict):
    print("### lungmask got parameters {}".format(param_dict))

    model_name   = param_dict["model_name"][0]
    rel_source_dir = param_dict["source_dir"][0]
    # remove trailing / at the beggining of name
    # otherwise os.path.join has unwanted behaviour for base dirs
    # i.e. join(/app/data_share, /app/wrongpath) = /app/wrongpath
    rel_source_dir = rel_source_dir.lstrip('/')

    log_debug("got model name {}".format(model_name))
    log_debug("got source dir {}".format(rel_source_dir))
    log_debug("calling get model")



    log_debug("Got model")
    data_share = os.environ["DATA_SHARE_PATH"]
    abs_source_dir = os.path.join(data_share, rel_source_dir)

    return run_lungmask_absolute(abs_source_dir, model_name=model_name)

def run_lungmask_absolute(abs_source_dir, model_name='R231CovidWeb'):

    data_share = os.environ["DATA_SHARE_PATH"]

    model = lungmask.get_model('unet', model_name)

    if not os.path.exists(abs_source_dir):
        log_debug("Input image source dir doesn't exist", abs_source_dir)
        return {}, False

    input_image = utils.get_input_image(abs_source_dir)
    segmentation = lungmask.apply(input_image, model, force_cpu=False, batch_size=20, volume_postprocessing=False)

    log_debug("Got result")
    log_debug(segmentation)

    result_dict = {}

    lungmask_dir = "lungmask_output"
    os.makedirs(os.path.join(data_share, lungmask_dir), exist_ok=True)

    time_now = str(time.time())
    hostname = os.environ["LUNGMASK_HOSTNAME"]

    rel_seg_save_path = os.path.join(lungmask_dir, f"{hostname}-segmentation-{time_now}.nii.gz")
    abs_seg_save_path = os.path.join(data_share, rel_seg_save_path)
    segmentation_nifti = sitk.GetImageFromArray(segmentation)
    segmentation_nifti.SetSpacing(input_image.GetSpacing())
    sitk.WriteImage(segmentation_nifti, abs_seg_save_path)

    result_dict["segmentation"] = rel_seg_save_path

    rel_input_save_path = os.path.join(lungmask_dir, f"{hostname}-input-{time_now}.nii.gz")
    abs_input_save_path = os.path.join(data_share, rel_input_save_path)

    sitk.WriteImage(input_image, abs_input_save_path)

    result_dict["input"] = rel_input_save_path

    return result_dict, True


if __name__ == "__main__":

    setup_logging()
    log_info("Started listening")

    served_requests = {
        "/lungmask_segment": run_lungmask
    }

    listener_server.start_listening(served_requests, multithreaded=True, mark_as_ready_callback=mark_yourself_ready)

