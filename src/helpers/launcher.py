import src.params
import subprocess


def launch():
    subprocess.call(["python3 server/client.py" +
                     " --input_file=" + src.params.message_sample_path +
                     " --output_file=" + src.params.output_sample_path +
                     " --srv_hostname=" + src.params.server_hostname +
                     " --srv_port=" + str(src.params.server_port)],
                    shell=True)


if __name__ == "__main__":
    launch()
