# MLD4ROMS
Potential density and curvature-based algorithm to compute the mixed layer depth diagnostic from the ocean models output such as ROMS, written in Python.

If you going to use this code and publish results in a journal, we request to cite the paper, where this method was introduced:
Osipov et al., 2017, Regional effects of the Mount Pinatubo eruption on the Middle East and the Red Sea, JGR Oceans.

The main.py and sample_data.nc files provide a minimal example based on the ROMS output to illustrate the algorithm. You will need to modify it to derive the mixed layer depth (MLD) based on your model output.

Things to consider:
1. The algorithm is likely sensitive to the vertical grid. In this example the grid is heavily weighted to the surface layers and quite sparse closer to the bottom.
2. The main.py reads the data from the netcdf, derives the depth and potential density (from the potential temperature and salinity using seawater library, install it if you don't have it). In your case you might have this data in the model output.
3. You could also apply this method to temperature or salinity, just by replacing the first argument, however, it wasn't tested.
4. Script provides some diagnostics regarding the method performance. Once you apply the algorithm to your data, I suggest to use and understand this diagnostics. Tuning will likely be required.
