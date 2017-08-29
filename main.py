from scipy.signal._peak_finding import _boolrelextrema
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import seawater as csr

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

# the code has been ported and modified from the MATLAB source code provided by Katja Lorbacher & Dietmar Dommenget
# their original paper :
# @Article{Lorbacher2006MLDCurvature,
#   Title                    = {Ocean mixed layer depth: A subsurface proxy of ocean-atmosphere variability},
#   Author                   = {Lorbacher, K. and Dommenget, D. and Niiler, P. P. and Köhl, A.},
#   Journal                  = {Journal of Geophysical Research: Oceans},
#   Year                     = {2006},
#   Url                      = {http://dx.doi.org/10.1029/2003JC002157}
# }


#these 2 routines below are just the way the depth is computed in ROMS model
#in your model it could be different and likely the depth is just saved as output
def get_roms_nc_parameters(nc_file_path):
    netcdf_dataset_impl = netCDF4.Dataset
    if ( type(nc_file_path) is list):
        netcdf_dataset_impl = netCDF4.MFDataset
    nc = netcdf_dataset_impl(nc_file_path)
    Vtransform = nc['Vtransform'][:]
    Vstretching = nc['Vstretching'][:]
    sc_r = nc['s_rho'][:]
    Cs_r = nc['Cs_r'][:]
    sc_w = nc['s_w'][:]
    Cs_w = nc['Cs_w'][:]
    return Vtransform, Vstretching, sc_r, Cs_r, sc_w, Cs_w
def compute_depths(nc_file_path, tyxSlices, igrid=1, idims=0):
    Vtransform, Vstretching, sc_r, Cs_r, sc_w, Cs_w = get_roms_nc_parameters(nc_file_path)
    netcdf_dataset_impl = netCDF4.Dataset
    if (type(nc_file_path) is list):
        netcdf_dataset_impl = netCDF4.MFDataset
    nc = netcdf_dataset_impl(nc_file_path)
    # Read in S-coordinate parameters.
    N = len(sc_r)
    Np = N + 1

    if (len(sc_w) == N):
        sc_w = np.cat(-1, sc_w.transpose())
        Cs_w = np.cat(-1, Cs_w.transpose())

    # Get bottom topography.
    yxSlice = [tyxSlices[1], tyxSlices[2]]
    h = nc.variables['h'][yxSlice]
    [Mp, Lp] = h.shape
    L = Lp - 1
    M = Mp - 1

    # Get free-surface
    zeta = nc.variables['zeta'][tyxSlices]
    # zeta=np.zeros([Lp, Mp])

    if igrid == 1:
        if idims == 1:
            h = h.transpose()
            zeta = zeta.transpose()
    elif igrid == 2:
        hp = 0.25 * (h[1:L, 1:M] + h[2:Lp, 1:M] + h[1:L, 2:Mp] + h[2:Lp, 2:Mp])
        zetap = 0.25 * (zeta[1:L, 1:M] + zeta[2:Lp, 1:M] + zeta[1:L, 2:Mp] + zeta[2:Lp, 2:Mp])
        if idims:
            hp = hp.transpose()
            zetap = zetap.transpose()
    elif igrid == 3:
        hu = 0.5 * (h[1:L, 1:Mp] + h[2:Lp, 1:Mp])
        zetau = 0.5 * (zeta[1:L, 1:Mp] + zeta[2:Lp, 1:Mp])
        if idims:
            hu = hu.transpose()
            zetau = zetau.transpose()
    elif igrid == 4:
        hv = 0.5 * (h[1:Lp, 1:M] + h[1:Lp, 2:Mp])
        zetav = 0.5 * (zeta[1:Lp, 1:M] + zeta[1:Lp, 2:Mp])
        if idims:
            hv = hv.transpose()
            zetav = zetav.transpose()
    elif igrid == 5:
        if idims:
            h = h.transpose()
            zeta = zeta.transpose()

    # Set critical depth parameter.
    hc = np.min(h[:])
    if (nc.variables.has_key('hc')):
        hc = nc['hc'][:]
    # Compute depths, for a different variables size will not match

    if (Vtransform == 1):
        if igrid == 1:
            for k in range(N):
                z0 = (sc_r - Cs_r) * hc + Cs_r(k) * h
                z = z0 + zeta * (1.0 + z0 / h)
                #     end
                #   case 2
                #     for k=1:N,
                #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hp;
                #       z(:,:,k)=z0 + zetap.*(1.0 + z0./hp);
                #     end
                #   case 3
                #     for k=1:N,
                #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hu;
                #       z(:,:,k)=z0 + zetau.*(1.0 + z0./hu);
                #     end
                #   case 4
                #     for k=1:N,
                #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hv;
                #       z(:,:,k)=z0 + zetav.*(1.0 + z0./hv);
                #     end
                #   case 5
                #     z(:,:,1)=-h;
                #     for k=2:Np,
                #       z0=(sc_w(k)-Cs_w(k))*hc + Cs_w(k).*h;
                #       z(:,:,k)=z0 + zeta.*(1.0 + z0./h);
                #     end
                # end
    elif Vtransform == 2:
        if igrid == 1:
            # we add time as 0 dimension to S(sigma, y, x) to get S(time, sigma, y, x)
            S = (hc * sc_r[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * Cs_r[np.newaxis, :, np.newaxis, np.newaxis]) / (hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
        elif igrid == 4:
            S = (hc * sc_r[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * Cs_r[np.newaxis, :, np.newaxis, np.newaxis]) / (hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
        elif igrid == 5:
            S = (hc * sc_w[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * Cs_w[np.newaxis, :, np.newaxis, np.newaxis]) / (hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
            #   case 2
            #     for k=1:N,
            #       z0=(hc.*sc_r(k)+Cs_r(k).*hp)./(hc+hp);
            #       z(:,:,k)=zetap+(zetap+hp).*z0;
            #     end,
            #   case 3
            #     for k=1:N,
            #       z0=(hc.*sc_r(k)+Cs_r(k).*hu)./(hc+hu);
            #       z(:,:,k)=zetau+(zetau+hu).*z0;
            #     end,
            #   case 4
            #     for k=1:N,
            #       z0=(hc.*sc_r(k)+Cs_r(k).*hv)./(hc+hv);
            #       z(:,:,k)=zetav+(zetav+hv).*z0;
            #     end,
            #   case 5
            #     for k=1:Np,
            #       z0=(hc.*sc_w(k)+Cs_w(k).*h)./(hc+h);
            #       z(:,:,k)=zeta+(zeta+h).*z0;
            #     end
            # end

    return z

#input is assumed to follow roms convention, i.e., from bottom to top
#       z = depth in meters, negative, from bottom to top
#       sigma = Potential density in kg m ^ {-3}, don\t forget to subtract 1000.
#       qi_treshold = treshold value for the quality index. If QI is less than this value, algorithm chooses largest first derivative
#output:
#       mld_data = mixed layer depth
#       below are the diagnostic variables that are produced by the algorithm:
#       sigma_gradient
#       sigma_curvature
#       find_ind
#       maxima_ind
#       qi_data
def compute_mld_based_on_density_curvature(sigma, z, qi_treshold=0.55):
    # z is assumed to follow roms convention, negative, from bottom to top
    # modified from the original idea by
    # Ocean mixed layer depth: A subsurface proxy of ocean-atmosphere variability, K. Lorbacher, D. Dommenget, P. P. Niiler, A. Kohl, 12 July 2006, DOI: 10.1029/2003JC002157

    sigma_sorted = sigma[::-1]
    z_sorted = -1 * z[::-1]
    dz = np.diff(z_sorted, axis=0)
    # forward difference
    sigma_gradient = np.diff(sigma_sorted, axis=0) / np.diff(z_sorted, axis=0)
    sigma_curvature = np.empty(sigma_gradient.shape)
    # backward difference
    sigma_curvature[1:] = np.diff(sigma_gradient, axis=0) / np.diff(z_sorted[:-1], axis=0)
    #at the edge have to use forward difference
    sigma_curvature[0] = np.diff(sigma_gradient[0:2], axis=0) / np.diff(z_sorted[0:2], axis=0)

    # find curvature local maxima
    curvature_extrema_max_ind = _boolrelextrema(sigma_curvature, np.greater, order=1, axis=0)
    #find positive first derivative and convex down
    ind_positive_derivative_positive_curvature = np.logical_and(sigma_gradient > 0, sigma_curvature > 0)
    #keep only correct maxima
    ind_max_extremum = np.logical_and(curvature_extrema_max_ind, ind_positive_derivative_positive_curvature)

    qi_data = np.empty(ind_max_extremum.shape)
    qi_data[:] = np.NaN

    # ind = np.where(ind_max_extremum)
    # ind = np.asarray(ind)
    # ind2 = np.minimum((1.5 * ind[0]).astype(int), z_sorted.shape[0])
    # t1 = time.time()
    # print 'start for loop ' + str(ind.shape[1])
    # for i in range(ind.shape[1]):
    #     qi_data[ind[0,i], ind[1,i], ind[2,i]] = 1 - np.std(sigma_sorted[0:ind[0,i], ind[1,i], ind[2,i]]) / np.std(sigma_sorted[0:ind2[i], ind[1,i], ind[2,i]])
    # print 'end for loop'
    # t2 = time.time()
    # print t2-t1

    #compute quality index everywhere
    for i in range(ind_max_extremum.shape[0]):
        qi_data[i] = 1 - np.std(sigma_sorted[0:i], axis=0) / np.std(sigma_sorted[0:min(i*1.5, ind_max_extremum.shape[0])], axis=0)

    masked_qi_data = np.ma.array(qi_data, mask=np.logical_not(ind_max_extremum))

    final_ind = np.empty(sigma_sorted.shape[1:])
    final_ind[:] = np.NaN

    profile_max_qi_data = np.nanmax(masked_qi_data, axis=0)
        # sub_index =  profile_max_qi_data < 0.3
        # # if qi index is small, then stratification is very weak, then pick local extremum/maxima which has bigest curvature maxima
        # masked_data = np.ma.array(sigma_curvature, mask=np.logical_not(ind_max_extremum))
        # final_ind[sub_index] = np.argmax(masked_data[:, sub_index], axis=0)
    # final_ind[sub_index] = np.argmax(sigma_curvature[:, sub_index], axis=0)
    sub_index = profile_max_qi_data >= qi_treshold
    #otherwise pick local extremum/maxima which has highest qi index
    final_ind[sub_index] = np.nanargmax(masked_qi_data[:, sub_index], axis=0)
    # if there is no local extremum or qi is weak, then pick point with max gradient value
    sub_index = np.logical_or(np.isnan(profile_max_qi_data), profile_max_qi_data < qi_treshold)
    final_ind[sub_index] = np.nanargmax(sigma_gradient[:, sub_index], axis=0)

    final_ind = final_ind.astype(int)
    fancy_indices = np.indices(ind_max_extremum.shape[1:])
    mld = z_sorted[final_ind, fancy_indices[0], fancy_indices[1]]

    # if (plot_diags):
    #     fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 18))
    #     plt.sca(axes[0])
    #     plt.plot(z_sorted[final_ind], sigma_sorted[final_ind], 'g*', markersize=20)
    #     plt.plot(z_sorted, sigma_sorted, 'k-o')
    #     plt.plot(z_sorted[ind], sigma_sorted[ind], 'ro')
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('sigma')
    #     plt.sca(axes[1])
    #     plt.plot(z_sorted[:-1][final_ind], sigma_gradient[final_ind], 'g*', markersize=20)
    #     plt.plot(z_sorted[:-1], sigma_gradient, 'k-o')
    #     plt.plot(z_sorted[:-1][ind], sigma_gradient[ind], 'ro')
    #     plt.gca().set_yscale('symlog', linthreshy=10 ** -5)
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('derivative')
    #     plt.sca(axes[2])
    #     plt.plot(z_sorted[:-1][final_ind], sigma_curvature[final_ind], 'g*', markersize=20)
    #     plt.plot(z_sorted[:-1], sigma_curvature, 'k-o')
    #     plt.plot(z_sorted[:-1][ind], sigma_curvature[ind], 'ro')
    #     plt.gca().set_yscale('symlog', linthreshy=10 ** -8)
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('curvature')
    #
    #     plt.sca(axes[3])
    #     plt.plot(z_sorted[:-1][ind], qi_data, 'ko')
    #     # plt.plot(z_sorted[:-1][curvature_extrema_max_ind], current_sigma_curvature[curvature_extrema_max_ind], 'ro')
    #     # plt.gca().set_yscale('symlog', linthreshy=10 ** -8)
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('qi')

    #invert in z order everything back to original and make compatible shape

    current_sigma_gradient_temp = np.zeros(sigma_sorted.shape)
    current_sigma_gradient_temp[:-1] = sigma_gradient
    current_sigma_gradient_temp[-1] = np.NaN
    sigma_gradient = current_sigma_gradient_temp[::-1]

    current_sigma_curvature_temp = np.zeros(sigma_sorted.shape)
    current_sigma_curvature_temp[:-1] = sigma_curvature
    current_sigma_curvature_temp[-1] = np.NaN
    sigma_curvature = current_sigma_curvature_temp[::-1]

    ind_max_extremum_temp = np.zeros(sigma_sorted.shape, dtype=bool)
    ind_max_extremum_temp[:-1] = ind_max_extremum
    ind_max_extremum_temp[-1] = np.NaN
    ind_max_extremum = ind_max_extremum_temp[::-1]

    qi_data_temp = np.zeros(sigma_sorted.shape)
    qi_data_temp[:-1] = qi_data
    qi_data_temp[-1] = np.NaN
    qi_data = qi_data_temp[::-1]

    return mld, sigma_gradient, sigma_curvature, final_ind, ind_max_extremum, qi_data


plt.ion()

#the file below is just a sample of the model output (cut down) to demonstrate the method, it is provided on the github
#MAKE SURE TO UPDATE THE FILE PATH TO THE RIGHT ONE
netcdf_file_path = './sample_data.nc'
# netcdf_file_path = '/shaheen/project/k1090/osipovs/Temp/github/sample_data.nc'

nc = netCDF4.Dataset(netcdf_file_path)
mask_rho = nc.variables['mask_rho'][:]

x_slice = slice(0, mask_rho.shape[1])
y_slice = slice(0, mask_rho.shape[0])

t_slice = slice(0, 1)
# t_slice = slice(50, 51)
tyxSlicesArray = [t_slice, y_slice, x_slice]

z = compute_depths(netcdf_file_path, tyxSlicesArray)
temp_profile = nc.variables['temp'][t_slice,:, y_slice, x_slice]
salt_profile = nc.variables['salt'][t_slice,:, y_slice, x_slice]
z = z.squeeze()
temp_profile = temp_profile.squeeze()
salt_profile = salt_profile.squeeze()

sigma_profile = csr.dens(salt_profile, temp_profile, 0)-1000
mld_data, sigma_gradient, sigma_curvature, final_ind, maxima_ind, qi_data = compute_mld_based_on_density_curvature(sigma_profile, z, qi_treshold=0.55)


plt.close('all')
mld_contours = np.linspace(-300, 0, 256)
#that is the list of points to plot diagnostics for
points_list = ((105, 75),(70, 180), (63, 100), (40, 220))
for i in range(len(points_list)):
    current_point = points_list[i]
    x_slice = slice(current_point[0], current_point[0] + 1)
    y_slice = slice(current_point[1], current_point[1] + 1)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))
    ax = axes[0]
    plt.sca(ax)
    im = plt.contourf(-1 * mld_data, levels=mld_contours, cmap='jet_r', extend='both')
    # plt.colorbar(aspect=10)#shrink=0.25)
    plt.contour(mask_rho, colors='k')
    plt.plot(x_slice.start, y_slice.start, 'wo', markersize=10, markerfacecolor='none', markeredgecolor='white', markeredgewidth=2)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    plt.title('MLD')

    cax = fig.add_axes([0.0, 0.1, 0.03, 0.8])
    plt.colorbar(im, cax=cax)

    current_temp = temp_profile[:, y_slice, x_slice].squeeze()
    current_salt = salt_profile[:, y_slice, x_slice].squeeze()
    current_z = z[:, y_slice, x_slice].squeeze()
    current_mld_2 = mld_data[y_slice, x_slice].squeeze()
    current_sigma = sigma_profile[:, y_slice, x_slice].squeeze()
    current_max_ind = maxima_ind[:, y_slice, x_slice].squeeze()
    current_sigma_gradient = sigma_gradient[:, y_slice, x_slice].squeeze()
    current_sigma_curvature = sigma_curvature[:, y_slice, x_slice].squeeze()
    current_qi = qi_data[:, y_slice, x_slice].squeeze()

    # current_mld_2, current_sigma_gradient, current_sigma_curvature, final_ind, maxima_ind = compute_mld_based_on_density_curvature(current_sigma, current_z, plot_diags=True)
    # current_mld *= -1
    # current_mld_2 *= -1
    # mld, qe, imf = get_mld_by_Lorbacher(current_z[::-1], current_temp[::-1])
    #
    print 'point ' + str(i) + ", is water " + str(mask_rho[y_slice, x_slice])
    ax = axes[1]
    plt.sca(ax)
    plt.plot(current_temp , current_z, 'b-o', label='temp')
    plt.xlabel('potential temperature')
    plt.ylabel('depth, (m)')
    plt.axhline(y=-1*current_mld_2, color='r', linewidth=2)
    plt.legend(loc='lower left')
    plt.ylim((np.min(current_z), 0))

    ax2 = ax.twiny()
    plt.sca(ax2)
    plt.plot(current_salt, current_z, 'r-o', label='salt')
    plt.xlabel('salinity')
    plt.legend(loc='lower right')
    plt.ylim((np.min(current_z), 0))

    ax3 = ax2.twiny()
    plt.sca(ax3)
    ax3.spines['top'].set_position(('axes', 1.08))
    ax2.spines['bottom'].set_position(('axes', 1))
    ax2.xaxis.set_ticks_position('top')
    plt.plot(current_sigma[current_max_ind], current_z[current_max_ind], 'g*', label='$\sigma$ d2f extremum', markersize=20)
    plt.plot(current_sigma, current_z, 'k-o', label='$\sigma$')
    plt.xlabel('$\sigma$')
    plt.legend(loc='lower center')
    plt.ylim((np.min(current_z), 0))

    formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax2.xaxis.set_major_formatter(formatter)
    ax3.xaxis.set_major_formatter(formatter)

    ax = axes[2]
    plt.sca(ax)
    plt.plot(current_sigma_gradient, current_z, 'k-o', label='d1f')
    plt.plot(current_sigma_gradient[current_max_ind], current_z[current_max_ind], 'ro')
    plt.gca().set_xscale('symlog', linthreshx=10 ** -5)
    plt.ylim((np.min(current_z), 0))
    plt.legend(loc='lower left')
    plt.xlabel('derivative')
    ax2 = ax.twiny()
    plt.sca(ax2)
    plt.plot(current_sigma_curvature, current_z, 'b-o', label='d2f')
    plt.plot(current_sigma_curvature[current_max_ind], current_z[current_max_ind], 'ro')
    plt.gca().set_xscale('symlog', linthreshx=10 ** -8)
    plt.ylim((np.min(current_z), 0))
    plt.legend(loc='lower right')
    plt.xlabel('curvature')

    ax = axes[3]
    plt.sca(ax)
    plt.plot(current_qi, current_z, 'ko')
    plt.plot(current_qi[current_max_ind], current_z[current_max_ind], 'ro')
    plt.ylim((np.min(current_z), 0))
    plt.title('quality index')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.1, wspace=0.2)


