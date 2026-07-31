import netCDF4 as nc
import numpy as np
import xarray as xr


def read_extent(topo_file, lon_min, lon_max, lat_min, lat_max):
    """读取指定经纬度范围的地形数据"""
    topoFile = nc.Dataset(topo_file, 'r')

    lons = topoFile.variables['x'][:]
    lats = topoFile.variables['y'][:]
    topo = topoFile.variables['z'][:]

    # 找到范围内的索引（注意处理经度从-180开始）
    lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]

    # 提取数据
    sub_lons = lons[lon_indices]
    sub_lats = lats[lat_indices]
    sub_topo = topo[np.ix_(lat_indices, lon_indices)]

    topoFile.close()

    return sub_lons, sub_lats, sub_topo

def save_ele(ele_path, lon_min, lon_max, lat_min, lat_max):
    etopo = xr.open_dataset(ele_path)
    etopo = etopo.rename({'x': 'lon', 'y': 'lat', 'z': 'elevation'})

    # 创建新的网格
    new_lon = np.arange(lon_min, lon_max, 0.25)
    new_lat = np.arange(lat_min, lat_max, 0.25)

    # 使用线性插值重采样
    etopo_resampled = etopo.interp(lon=new_lon, lat=new_lat, method='linear')

    ele = etopo_resampled.elevation.values
    print(ele.shape)

    np.save(save_ele_path, ele)

ele_path = '../source/ETOPO1_gmt4.grd'
save_ele_path = '../processed_ele/northeast_elevation.npy'
lon_min, lon_max = 124.75, 127.5
lat_min, lat_max = 44.75, 47.5
lons, lats, topo = read_extent(ele_path, lon_min, lon_max, lat_min, lat_max)
save_ele(ele_path, lon_min, lon_max, lat_min, lat_max)


print(f"经度范围: {lons.min()} ~ {lons.max()}, 点数: {len(lons)}")
print(f"纬度范围: {lats.min()} ~ {lats.max()}, 点数: {len(lats)}")
print(f"地形数据形状: {topo.shape}")
print(f"高程范围: {topo.min()} ~ {topo.max()}米")