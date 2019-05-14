import params
import subprocess

def launch_test(input_file, output_file):
    hostname = params.server_hostname
    port = params.server_port

    subprocess.call('python3 client.py --input_file '+input_file+ ' --output_file '+ output_file +
                    ' --srv_hostname '+ hostname + ' --srv_port '+port, shell=True)
 
launch_test("../data/input.txt", "../data/output.txt")