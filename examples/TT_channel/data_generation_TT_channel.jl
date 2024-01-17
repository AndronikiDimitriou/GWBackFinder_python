using Pkg
# Pkg.add("https://github....")
Pkg.activate("/home/zaldivar/Documents/Androniki/Github/GWBackFinder.jl") #to remove
using GWBackFinder
using ProgressMeter
using HDF5
using ArgParse

# define the arguments to be passed to the script via ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--input", "-i"
        help = "input parameters file path"
        arg_type = String
    "--output-dir", "-o"
        help = "output folder"
        arg_type = String
    "--dataset-name", "-d"
        help = "dataset name"
        arg_type = String
    "--size_dataset", "-f"
        help = "size_dataset"
        arg_type = Int64
end

parsed_args = parse_args(ARGS, s)

file_path = parsed_args["input"]
output_dir = mkpath(parsed_args["output-dir"])

# Open the HDF5 file
file = h5open(file_path, "r")

# Access the dataset by its name
dataset_name = parsed_args["dataset-name"]
thetas = read(file[dataset_name])

# Close the HDF5 file
close(file)


### Define the frequency range
f = range(start=3 * 1e-5, stop=0.5, step=1e-6) 

### Split in 26 bins. idx26 shows in which bin the frequency belongs to and logbins_26 are the boundaries of the bins . After we coarse grain the frequencies from f = 10−3 Hz to the maximum frequency
### fmax = 0.5 Hz), i.e., we bin them in 1000 intervals of equal log-spacing. idx shows in which of the 1000 bins the frequency belongs to and logbins the boundaries of the 1000 bins.
idx,idx26,logbins_26,logbins,f_filtered=GWBackFinder.binning(f)


### Generate and save data
@showprogress for i in 1:parsed_args["size_dataset"]
    Data_total= GWBackFinder.model_noise_train_data(f,transpose(thetas)[i,:])
    GWBackFinder.write_sample(Data_total,"$(output_dir)/$(i-1).jld2")
end
