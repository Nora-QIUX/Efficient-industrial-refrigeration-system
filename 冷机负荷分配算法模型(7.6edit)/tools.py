import pandas as pd
import json
import requests
import psycopg2

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

# 创建时间戳


def get_date(start_time, end_time, range=1):  # 生成自增时间戳
    tmp_range = pd.period_range(start=start_time, end=end_time,
                                freq='{}t'.format(range), )
    tmp_frame = pd.DataFrame()
    tmp_frame['data_time'] = tmp_range
    # tmp_frame['data_time'] = tmp_frame['data_time'].astype('str')
    # tmp_frame['data_time'] = tmp_frame['data_time']. \
    #     apply(
    #     lambda x: datetime.strptime(
    #         x,
    #         '%Y-%m-%d %H:%M').strftime("%Y-%m-%d %H:%M:%S")).astype('str')
    return tmp_frame

# 获取冷量数据


def get_cold_data(start_time, end_time, system_id):
    url = "http://106.14.227.58:8098/base/get/external"
    payload = json.dumps({
        "property": 2,
        "systemId": system_id,
        # 此处需随项目要求更改，local/all/dech，如果不写，默认是取辅助优先级为1的值 [Nora]
        "systemName":'local',
        "startTime": start_time,
        "endTime": end_time
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res_dict = json.loads(response.text).get('data')
    return res_dict

# 获取电量


def get_power_data(start_time, end_time, system_id):
    url = "http://106.14.227.58:8098/base/get/external"
    payload = json.dumps({
        "property": 1,
        "systemId": system_id,
        "systemName": 'all',
        "startTime": start_time,
        "endTime": end_time
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res_dict = json.loads(response.text).get('data')
    return res_dict

# 获取流量

def get_flow_data_api(start_time, end_time, system_id, system_name, device_name):
    url = "http://106.14.227.58:8098/base/get/external"
    payload = json.dumps({
        "property": 3,
        "systemId": system_id,
        "systemName": system_name,
        "startTime": start_time,
        "endTime": end_time
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res_dict = json.loads(response.text).get('data')
    flow = res_dict.get('part').get(device_name)
    flow = 0 if flow==None else flow
    return flow

def get_heat_data(start_time, end_time, system_id):
    url = "http://106.14.227.58:8098/base/get/external"
    payload = json.dumps({
        "property": 4,
        "systemId": system_id,
        "systemName": 'dech',
        "startTime": start_time,
        "endTime": end_time
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res_dict = json.loads(response.text).get('data')
    return res_dict

