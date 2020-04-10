import subprocess as sb
import os
from http.server import *
from urllib.parse import urlparse
from urllib.parse import parse_qs
import logging
import sys
from lungmask import lungmask
from lungmask import utils
import SimpleITK as sitk
import numpy as np
import json
import time

class CommandRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text")
        self.end_headers()

        self.__requested_method = {
            "/lungmask_segment": run_lungmask
        }


    def do_GET(self):
        self._set_headers()
        self.__handle_request()

    def __handle_request(self):
        parsed_url = urlparse(self.path)
        parsed_params = parse_qs(parsed_url.query)

        log_debug("Got request with url {} and params {}".format(parsed_url.path, parsed_params))

        if parsed_url.path not in self.__requested_method:
            log_debug("unkown request {} received".format(self.path))
            return

        # called lungmask

        print("running lungmask")
        result_dict = self.__requested_method[parsed_url.path](parsed_params)
        # serialised_result = json.dumps(result.tolist())
        # self.wfile.write(serialised_result.encode())
        print("result", result_dict)

        print("sending over", result_dict)
        self.wfile.write(json.dumps(result_dict).encode())



def run(server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)

    mark_yourself_ready()
    httpd.serve_forever()


def mark_yourself_ready():
    hostname = os.environ['HOSTNAME']
    data_share_path = os.environ['DATA_SHARE_PATH']
    cmd = "touch {}/{}_ready.txt".format(data_share_path, hostname)

    logging.info("Marking as ready")
    sb.call([cmd], shell=True)


def run_lungmask(param_dict):
    print("### lungmask got parameters {}".format(param_dict))

    model_name   = param_dict["model_name"][0]
    download_dir = param_dict["source_dir"][0]

    log_debug("got model name {}".format(model_name))
    log_debug("got source dir {}".format(download_dir))
    log_debug("calling get model")
    model = lungmask.get_model('unet', model_name)

    log_debug("Got model")
    data_share = os.environ["DATA_SHARE_PATH"]
    abs_source_dir = os.path.join(data_share, download_dir)
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

    return result_dict

def setup_logging():
    file_handler = logging.FileHandler("log.log")
    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=logging.debug, # TODO level=get_logging_level(),
        # format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            file_handler,
            stream_handler
        ]
    )

def log_info(msg):
    logging.info(msg)

def log_debug(msg):
    logging.debug(msg)

def log_warning(msg):
    logging.warning(msg)

def log_critical(msg):
    logging.critical(msg)


if __name__ == '__main__':
    setup_logging()
    log_info("Started listening")
    run(handler_class=CommandRequestHandler)

    # model = lungmask.get_model('unet', model_name)
    # input_image = utils.get_input_image(download_dir)
    # input_nda = sitk.GetArrayFromImage(input_image)
    # print(input_nda.shape)
    # zd, yd, xd = input_nda.shape
    #
    # spx, spy, spz = input_image.GetSpacing()
    # return lungmask.apply(input_image, model, bar2, force_cpu=False, batch_size=20, volume_postprocessing=False)

# def run_lungmask():
#     cmd = "lungmask /home/source /home/output/{} ".format(os.environ["OUTPUT_NAME"])
#
#     if "MODEL" in os.environ:
#         cmd += " --modelname {}".format(os.environ["MODEL"])
#
#     sb.call([cmd], shell=True)

#
# if __name__ == "__main__":
#     run_lungmask()