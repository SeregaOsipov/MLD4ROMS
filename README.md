# MLD4ROMS
Potential density and curvature-based algorithm to compute the mixed layer depth diagnostic from the ocean models output such as ROMS, written in Python.

If you are going to use this code and publish the results in a scientific journal, we request to cite the paper, where this code was introduced:
Osipov et al., 2017, Regional effects of the Mount Pinatubo eruption on the Middle East and the Red Sea, JGR Oceans.


Make sure that you have all the libraries installed (check the imports section). These are numpy, matplotlib. For ROMS model output, script relies on seawater library (version > 3, https://pypi.python.org/pypi/seawater) to compute potential density from potential temperature and salinity.

The main.py and sample_data.nc files provide a minimal example based on the ROMS output to illustrate the algorithm. You will need to modify it to derive the mixed layer depth (MLD) based on your model output.

Things to consider:
1. The algorithm is likely sensitive to the vertical grid. In this example the grid is heavily weighted to the surface layers and quite sparse closer to the bottom.
2. The main.py reads the data from the netcdf file, derives the depth and potential density (from the potential temperature and salinity using seawater library, install it if you don't have it). In your case you might have these data in the model output.
3. You could also apply this method to temperature or salinity, just by replacing the first argument, however, it wasn't tested.
4. Script provides some diagnostics regarding the method performance (such as quality index) and visualizes the profiles. Once you apply the algorithm to your data, I suggest to use and understand these diagnostics. Tuning will likely be required.
