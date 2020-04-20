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

    model = lungmask.get_model('unet', model_name)


    log_debug("Got model")
    data_share = os.environ["DATA_SHARE_PATH"]
    abs_source_dir = os.path.join(data_share, rel_source_dir)

    if not os.path.exists(abs_source_dir):
        log_debug("Input image source dir doesn't exist", abs_source_dir)
        return {}, False

    input_image = utils.get_input_image(abs_source_dir)

    segmentation = lungmask.apply(input_image, model, force_cpu=False, batch_size=20, volume_postprocessing=False)

    log_debug("Got result")
    log_debug(segmentation)

    result_dict = {}

    time_now = str(time.time())

    # saving segmentation to file
    rel_seg_save_path = "{}-segmentation-{}".format(os.environ["LUNGMASK_HOSTNAME"], time_now)
    seg_save_path = os.path.join(data_share, rel_seg_save_path)

    print("saving np array to", seg_save_path)
    np.save(seg_save_path, segmentation)

    result_dict["segmentation"] = rel_seg_save_path + ".npy"

    # saving input image as numpy array to file
    input_nda = sitk.GetArrayFromImage(input_image)
    rel_input_save_path = "{}-input-nda-{}".format(os.environ["LUNGMASK_HOSTNAME"], time_now)
    input_save_path = os.path.join(data_share, rel_input_save_path)
    np.save(input_save_path, input_nda)

    result_dict["input_nda"] = rel_input_save_path + ".npy"

    # send input_image spacing
    spx, spy, spz = input_image.GetSpacing()
    result_dict["spacing"] = (spx, spy, spz)

    return result_dict, True


if __name__ == "__main__":

    setup_logging()
    log_info("Started listening")

    served_requests = {
        "/lungmask_segment": run_lungmask
    }

    listener_server.start_listening(served_requests, multithreaded=True, mark_as_ready_callback=mark_yourself_ready)

