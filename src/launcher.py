import params
import subprocess


def launch():
    subprocess.call(["python3 client.py" +
                     " --input_file=" + params.input_sample_file_path +
                     " --output_file=" + params.output_sample_file_path +
                     " --srv_hostname=" + params.server_hostname +
                     " --srv_port=" + str(params.server_port)],
                    shell=True)


if __name__ == "__main__":
    launch()
