# Run this program from the project root

import os

sources_path = "src/main/resources/net/javaman/flowkey/cuda/"
sources = [
    "Util.cu",
    "InitialComparison.cu",
    "NoiseReduction.cu",
    "FlowKey.cu"
]
source_output_path = "ptx_out/"
source_output = "Source.cu"
ptx_output = "Source.ptx"

source = ""
for filename in sources:
    with open(sources_path + filename) as f:
        source += f.read() + "\n\n"

if not os.path.exists(source_output_path):
    os.makedirs(source_output_path)
with open(source_output_path + source_output, "w+") as f:
    f.write(source)

os.system("nvcc -ptx \"" + source_output_path + source_output + "\" -o \"" + source_output_path + ptx_output + "\"")