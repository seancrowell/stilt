#========================
# sample_wrfout.py
#
# purpose: samples wrfout files to pull out boundary conditions for STILT trajectories
# author: Sean Crowell
# date: Sept 19, 2024
# input: boundary point HDF file from compile_boundary_points.py script
# output: boundary point HDF file augmented with wrfout samples
#========================

from h5py import File
import netCDF4 as nc
import glob,sys,os
import numpy as np
import datetime as dt
import scipy
from pykdtree.kdtree import KDTree

domain = 'd02'

fname = 'bnd_loc/20230726_F1/202307261333_-74.5701_40.3034_10000_traj_bnd.h5'
with File(fname,'r') as loc_f:
    loc_f = File(fname,'r')
    alt = loc_f['particle_altitude'][:]
    lon = loc_f['particle_longitude'][:]
    lat = loc_f['particle_latitude'][:]
    t = loc_f['particle_time'][:]
    obs_t = (dt.datetime.strptime(loc_f.attrs['obs_time'][:],'%Y-%m-%d %H:%M:%S UTC')-dt.datetime(1970,1,1)).total_seconds()
    loc_f.close()
#==================================================
# This stuff will need to be changed when we have Steve's files

wrfghg_prefix = '/scratch/01178/tg804586/Run/CO2_and_otherGHG/WRFV4.5.2/CONUS/wrfchem4.5.2_Hu2021JGR_CH4NEI2017_useIndividualWrfinput_NYcity.'
# adjustment for lack of files in Xiao-Ming's directories
actual_part_t = np.array([dt.datetime(1970,1,1) + dt.timedelta(seconds=ti) for ti in t])
part_t = np.array([a+dt.timedelta(days=366) for a in actual_part_t])
# runs are reinitialized every 6 hours and run for 18
hr = [0,6,12,18,24]
# pick the right directory by hour number
part_h = np.array([ti.hour for ti in part_t])
inds = np.digitize(part_h,hr)
wrfout_h = []
wrfout_dir_str = []
for ip,ind in enumerate(inds):
    if part_h[ip] == hr[ind-1]:
        wrfout_h.append(hr[ind-2])
    else:
        wrfout_h.append(hr[ind-1])
    wrfout_dir_str.append((part_t[ip]-dt.timedelta(seconds=3600*int(part_h[ip]-wrfout_h[ip]))).strftime('%Y%m%d%H'))
wrfout_h = np.array(wrfout_h)
# find all of the files in Xiao-Ming's directory that match up with the trajectories
wrf_files = np.array([wrfghg_prefix+wrfout_dir_str[ip]+'/wrfout_'+domain+f'_{part_t[ip].strftime('%Y-%m-%d_%H')}:00:00' for ip in range(len(part_t))])
# Only loop over the unique filenames to save I/O
unique_wrf_files = sorted(np.array(list(set(wrf_files))))
#===================================================

bc_lat = [] # NN lat of WRF model
bc_lon = [] # NN lon of WRF model
bc_z = []   # NN altitude of WRF model
bc_co2 = [] # CO2 at WRF 3D NN gridbox
bc_ch4 = [] # CH4 at WRF 3D NN gridbox
bc_t = [] # WRFOUT file time
for fi in unique_wrf_files:
    print(fi)
    f = nc.Dataset(fi,'r')
    wrf_lat = f['XLAT'][:][0]
    wrf_lon = f['XLONG'][:][0]
    wrf_p = (f['P'][:] + f['PB'][:])[0]
    wrf_z = wrf_p/9.8
    wrf_t = fi.split('_')[-1]

    tree = KDTree(np.c_[wrf_lon.ravel(),wrf_lat.ravel()])

    part_inds = np.where(wrf_files == fi)[0]
    part_z = alt[part_inds]
    part_lat = lat[part_inds]
    part_lon = lon[part_inds]
    part_points = np.float32(np.c_[part_lon,part_lat])

    dd,ii = tree.query(part_points,k=1,eps=0.0)
    lat_inds,lon_inds = np.unravel_index(ii,wrf_lat.shape,order='C')
    bc_lat.extend(wrf_lat[lat_inds,lon_inds])
    bc_lon.extend(wrf_lon[lat_inds,lon_inds])
    bc_t.extend([wrf_t for i in range(len(part_inds))])

    sub_z = wrf_z[:,lat_inds,lon_inds]
    z_inds = np.array([np.argmin((part_z[ip]-sub_z[:,ip])**2) for ip in range(len(part_inds))])
    bc_z.extend(wrf_z[z_inds,lat_inds,lon_inds])

    co2 = np.zeros(len(part_inds))
    for co2_v in ['ANT','BIO','OCE']:
        co2 += f['CO2_'+co2_v][0][z_inds,lat_inds,lon_inds]-f['CO2_BCK'][0][z_inds,lat_inds,lon_inds]
    co2 += f['CO2_BCK'][0][z_inds,lat_inds,lon_inds]
    bc_co2.extend(co2)

    ch4 = np.zeros(len(part_inds))
    for ch4_v in ['ANT','BIO','TST','BBU']:
        ch4 += f['CH4_'+ch4_v][0][z_inds,lat_inds,lon_inds]-f['CH4_BCK'][0][z_inds,lat_inds,lon_inds]
    ch4 += f['CH4_BCK'][0][z_inds,lat_inds,lon_inds]
    bc_ch4.extend(ch4)

with File(fname,'r') as loc_f:
    loc_f = File(fname,'r')
