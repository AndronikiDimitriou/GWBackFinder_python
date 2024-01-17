using Pkg
#Pkg.activate("./")
Pkg.activate("/home/zaldivar/Documents/Androniki/Github/GWBackFinder.jl")
using GWBackFinder
using HDF5
using ArgParse

### Define the frequency range

s = ArgParseSettings()
@add_arg_table s begin

    "--parameters", "-i"
        help = "input parameters file path"
        arg_type = String

    "--output-dir", "-o"
        help = "output folder"
        arg_type = String

    "--signal", "-s"
        help = "type of signal"
        arg_type = String

end

parsed_args = parse_args(ARGS, s)

signal = parsed_args["signal"]

file_path = parsed_args["parameters"]

# Open the HDF5 file
file = h5open(file_path, "r")

# Access the dataset by its name
parameters_signal = read(file["parameters_signal"])
parameters_noise = read(file["parameters_noise"])

# Close the HDF5 file
close(file)

parsed_args = parse_args(ARGS, s)

f = range(start=3 * 1e-5, stop=0.5, step=1e-6)

### Split in 26 bins. idx26 shows in which bin the frequency belongs to and logbins_26 are the boundaries of the bins . After we coarse grain the frequencies from f = 10−3 Hz to the maximum frequency
### fmax = 0.5 Hz), i.e., we bin them in 1000 intervals of equal log-spacing. idx shows in which of the 1000 bins the frequency belongs to and logbins the boundaries of the 1000 bins.

idx,idx26,logbins_26,logbins,f_filtered = GWBackFinder.binning(f)

### Define different signals


if signal=="powerlaw"
    z_powerlaw               =  GWBackFinder.zPowerLaw(parameters_signal)
    z_noise                  =  GWBackFinder.zNoise(parameters_noise)
    Data_AA,Data_TT,f_total_ΑΑ,f_total_TT  = GWBackFinder.different_signals(z_powerlaw,z_noise,f,f_filtered,logbins,idx)
 
elseif signal=="peak"
    z_peak                   =  GWBackFinder.zPeak(parameters_signal)
    z_noise                  =  GWBackFinder.zNoise(parameters_noise)
    Data_AA,Data_TT,f_total_ΑΑ,f_total_TT  = GWBackFinder.different_signals(z_peak,z_noise,f,f_filtered,logbins,idx)
    
elseif signal=="wiggly"
    z_wiggly                 =  GWBackFinder.zWiggly(parameters_signal)
    z_noise                  =  GWBackFinder.zNoise(parameters_noise)
    Data_AA,Data_TT,f_total_ΑΑ,f_total_TT  = GWBackFinder.different_signals(z_wiggly,z_noise,f,f_filtered,logbins,idx)
    
elseif signal=="bpowerlaw"
    z_Broken_powerlaw        =  GWBackFinder.zBroken_powerlaw(parameters_signal)
    z_noise                  =  GWBackFinder.zNoise(parameters_noise)
    Data_AA,Data_TT,f_total_ΑΑ,f_total_TT  = GWBackFinder.different_signals(z_Broken_powerlaw,z_noise,f,f_filtered,logbins,idx)
    
elseif signal=="dpeak"
    z_Double_peaks           =  GWBackFinder.zDouble_peaks(parameters_signal)
    z_noise                  =  GWBackFinder.zNoise(parameters_noise)
    Data_AA,Data_TT,f_total_ΑΑ,f_total_TT   = GWBackFinder.different_signals(z_Double_peaks,z_noise,f,f_filtered,logbins,idx)
    

elseif signal=="three_peaks"
    z_three_peaks            =  GWBackFinder.zThree_peaks(parameters_signal)
    z_noise                  =  GWBackFinder.zNoise(parameters_noise)
    Data_AA,Data_TT,f_total_ΑΑ,f_total_TT = GWBackFinder.different_signals(z_three_peaks,z_noise,f,f_filtered,logbins,idx)
    
else 
    println("Not correct type of signal")
    Base.exit()
end

output_file_path = parsed_args["output-dir"]

# Open the HDF5 file in write mode
h5open(output_file_path, "w") do file
    # Create a dataset and write the data
    write(file, "Data_AA",Data_AA)
    write(file, "Data_TT",Data_TT)
    write(file, "f_total_ΑΑ",f_total_ΑΑ)
    write(file, "f_total_TT",f_total_TT )
end