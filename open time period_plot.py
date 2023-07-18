import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import psycopg2
import matplotlib.ticker as ticker

# pg数据库连接
pgsql_dict = {
    'database': "discovery_000012",
    'username': "tsnyadmin",
    'password': "Tsny2020",
    'host': "pgm-0jl416j2noxs2x83bo.pg.rds.aliyuncs.com", 'port': "5432"
}
conn_000012 = psycopg2.connect(
    database='discovery_000012',
    user=pgsql_dict.get('username'),
    password=pgsql_dict.get('password'),
    host=pgsql_dict.get('host'),
    port="5432")

# mysql数据库连接
mysql_dict = {
    'localhost': "rm-uf6q54n1p7mx87ef61o.mysql.rds.aliyuncs.com",
    'username': "ts_admin",
    'password': "Tsadmin123",
    'database': "discovery_cloud",
}

def open_time_period_plot(system_id:str,start_time, end_time):
    sql_dech = '''
    select data_time, device_name, flow_instantaneous, temp_chw_in, temp_chw_out, power_active
    from dech_{}_l1
    where data_time between '{}' and '{}'
    '''.format(system_id, start_time, end_time)
    conn = conn_000012
    df_dech = pd.read_sql(sql_dech, conn)

    df_dech = df_dech.dropna().reset_index(drop=True)
    df_dech['data_time'] = pd.to_datetime(df_dech['data_time'])

    for i in ['dech01', 'dech02', 'dech03']:
        column_name = '{}_open'.format(i)
        df_dech[column_name] = (df_dech['device_name'] == i) & (df_dech['flow_instantaneous'] > 100) & (df_dech['power_active'] > 10)
        df_dech[column_name] = df_dech[column_name].astype(float)
        if i == 'dech02':
            df_dech[column_name] = df_dech[column_name] + 0.1
        if i == 'dech03':
            df_dech[column_name] = df_dech[column_name] + 0.2

    # 绘制图形
    fig, ax1 = plt.subplots()
    plt.ylim(-0.5, 1.5)
    for i in ['dech01_open', 'dech02_open', 'dech03_open']:
        ax1.scatter(df_dech['data_time'], df_dech[i], label=i, s=5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('open_or_not')
    ax1.legend(loc='center left')
    plt.title('Open Time Period')

    # 自动调整日期标签
    fig.autofmt_xdate()

    plt.show()

    return df_dech



if __name__ == "__main__":
    aa = open_time_period_plot('3051', start_time='2023-07-07 00:00:00', end_time='2023-07-11 00:00:00')