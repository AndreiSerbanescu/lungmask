import subprocess as sb
import argparse
import yaml
import os

def validate_output(output):

    split = os.path.split(output)

    if '.' not in split[1]:
        return output, "output.nii.gz"

    return split

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LungMask")

    parser.add_argument('-o', '--output', required=True, type=str, help="Host path to output directory or filename")
    parser.add_argument('-s', '--source', default="", type=str, help="Host path to source volume directory")
    parser.add_argument('-m', '--model', default="", type=str, help="Select model: only R231, LTRCLobes or R231CovidWeb")
    parser.add_argument('--debug', action='store_true', help="Select this for container to not perform any computation")

    args = parser.parse_args()

    if args.model != "" and args.model not in ["R231", "LTRCLobes", "R231CovidWeb"]:
        print("Incorrent model, please input R231, LTRCLobes or R231CovidWeb or do not use --model argument")
        exit()

    with open('docker-compose.yml') as docker_compose_file:

        dc = yaml.load(docker_compose_file)

        source = '{}:/home/source'.format(args.source)

        output_dir, output_filename = validate_output(args.output)

        output = '{}:/home/output'.format(output_dir)

        volumes = [source, output]

        dc['services']['lungmask']['volumes'] = volumes

        if args.debug:
            dc['services']['lungmask']['command'] = 'tail -F anything'

        environment = ["DEBUG={}".format(args.debug), "OUTPUT_NAME={}".format(output_filename)]

        # add MODEL environment variable
        if args.model != "":
            environment.append("MODEL={}".format(args.model))

        old_env = dc['services']['lungmask'].get('environment', [])
        dc['services']['lungmask']['environment'] = environment + old_env


    new_dc_file = 'docker-compose-tmp.yml'

    if os.path.exists(new_dc_file):
        os.remove(new_dc_file)

    with open(new_dc_file, 'w') as docker_compose_tmp_file:
        yaml.dump(dc, docker_compose_tmp_file)

    docker_compose_up_cmd = "docker-compose down --volumes " \
                            "&& docker-compose -f {} up --build".format(new_dc_file)
    sb.call([docker_compose_up_cmd], shell=True)

