import numpy as np

import params
import launcher
import plot_helper


if __name__ == "__main__":

    # Send the samples from the input file to the server, and get the output samples in the output file
    launcher.launch()

    # Load the input and output samples from their respective files
    input = np.loadtxt(params.input_sample_file_path)
    output = np.loadtxt(params.output_sample_file_path)

    # Plot the input and output samples in Time domain
    plot_helper.two_simple_plots(input, output, "Input and output in Time domain", "Input", "Output")

    # Plot the output samples in the Frequency domain
    plot_helper.two_fft_plots(input, output, "Input and output in Frequency domain", "Input", "Output")
