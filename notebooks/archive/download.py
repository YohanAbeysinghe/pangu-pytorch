import os
os.environ["CDSAPI_RC"] = "/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch/.cdsapirc"

import cdsapi
import argparse
from calendar import monthrange
def downloadERA5_surface(year, month, days):

    print("downloading..."+year+month+'.nc')
  
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf_legacy',
            'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'mean_sea_level_pressure',
        ],
            'year': year,
            'month': month,
            'day': days,
            # 'time': [
            #     '00:00', '01:00', '02:00',
            #     '03:00', '04:00', '05:00',
            #     '06:00', '07:00', '08:00',
            #     '09:00', '10:00', '11:00',
            #     '12:00', '13:00', '14:00',
            #     '15:00', '16:00', '17:00',
            #     '18:00', '19:00', '20:00',
            #     '21:00', '22:00', '23:00',
            # ],
            'time': [
                '00:00', '12:00',
            ],
        },
        os.path.join(args.output_dir,'surface', 'surface_'+year+month+'.nc'))

    
def downloadERA5_upper(year, month, day):
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
            'geopotential', 'specific_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
            'pressure_level': [
            '50', '100', '150',
            '200', '250', '300',
            '400', '500', '600',
            '700', '850', '925',
            '1000',
        ],
            'year': year,
            'month': month,
            'day': day,
            # 'time': [
            #     '00:00', '01:00', '02:00',
            #     '03:00', '04:00', '05:00',
            #     '06:00', '07:00', '08:00',
            #     '09:00', '10:00', '11:00',
            #     '12:00', '13:00', '14:00',
            #     '15:00', '16:00', '17:00',
            #     '18:00', '19:00', '20:00',
            #     '21:00', '22:00', '23:00',
            # ],
            'time': [
                '00:00', '12:00',
            ],
        },
        os.path.join(args.output_dir, 'upper', 'upper_'+year+month+day+'.nc'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default= "/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch/notebooks")
    args = parser.parse_args()

    # download 2015-2019 ERA5 surface variable
    years = range(2019,2020)
    months = range(1, 13)
    
    c = cdsapi.Client()
    for year in years:
       for month in months:
            num_days = monthrange(year, month)[1]
            days =[f"{i:02d}" for i in range(1, num_days + 1)]
            downloadERA5_surface(str(year), f"{month:02d}", days)