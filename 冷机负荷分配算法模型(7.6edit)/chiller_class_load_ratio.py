import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import json
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
# from tools import conn_000012
from sklearn.metrics import mean_absolute_percentage_error as mape_sklearn
import os
# os.chdir('C:/Users/User/PycharmProjects/algorithm-library-tansuo/qiuxin/冷机负荷分配算法模型(7.6edit)')  # 设置工作目录
from adtk.detector import QuantileAD, LevelShiftAD

from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from pyecharts import options as opts
from pyecharts.charts import Scatter3D
from pyecharts.commons.utils import JsCode


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

class Chiller(object):

    def __init__(self, system_id:str, device_name: str, Cap: float, P_ref: float, Delta_tch_des: float,
                 Vch_des: float):
        """
        Parameters
        ----------
        system_id :项目编号
        device_name :冷机编号
        Cap :冷机额定制冷量
        P_ref :冷机额定功率
        Delta_tch_des ：冷机冷冻侧设计进出口温差
        Vch_des ：冷机冷冻侧设计水流量
        -------
        """

        # =================一些常数==============================
        self.RT2KW = 3.517
        self.cpw = 4.2
        self.pw = 1000  # 水的密度,kg/m3

        # ==================冷机型号=============================
        self.system_id = system_id
        self.device_name = device_name
        self.Cop_ref = Cap / P_ref * self.RT2KW  # US RT to kW
        self.P_ref = P_ref
        self.Cap = Cap * self.RT2KW  # USRT to kw
        self.Delta_tch_des = Delta_tch_des  # 蒸发器进出水设计温差
        self.Vch_des = Vch_des * 3.6  # L/s to m3/h
        # 上下四分位数筛选方法，参数设定
        self.quantile_ad = QuantileAD(high=0.9, low=0.1)
        # 根据突变斜率筛选方法，参数设定
        self.level_shift_ad = LevelShiftAD(c=1.0, side='both', window=20)


    #####################################Power开始########################################################
    # Question:和薛博士的power计算逻辑不一样？有什么选择的说法嘛？
    # Answer:因为两者主要目的不一样，但后期会整合这两个出来一个更好的。
    def ratio_load_verify(self, start_time, end_time):
        """
        Parameters
        X :[tchin, tchout, Vch, tcdin]
        tchin :冷机冷冻水进水温度
        tchout :冷机冷冻水出水温度
        Vch :冷机冷冻水流量
        tcdin :冷机冷却水进水温度
        """
        sql_dech = '''
        select data_time, temp_chw_in, temp_chw_out, flow_instantaneous, power_active, ratio_load
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        '''.format(self.system_id, self.device_name, start_time, end_time)

        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)
        df_dech['CL'] = df_dech['flow_instantaneous'] * (
                    df_dech['temp_chw_in'] - df_dech['temp_chw_out']) * 4.1868 * 1000 / 3600
        df_dech['ratio_load_verified'] = df_dech['CL']/self.Cap
        df_dech['ratio_load'] /= 100
        df_dech['ratio_load_diff'] = abs(df_dech['ratio_load'] - df_dech['ratio_load_verified'])

        detect_data = pd.DataFrame()
        detect_data.index = df_dech['data_time']
        detect_data['power_active'] = df_dech['power_active'].values
        detect_data['flag_1'] = self.level_shift_ad.fit_detect(detect_data['power_active'])
        detect_data['flag_2'] = self.quantile_ad.fit_detect(detect_data['power_active'])
        detect_data = detect_data.reset_index(drop=False)
        # 这里会自动对齐正确的index
        df_dech = df_dech[(detect_data['flag_1'] == False) &
                          (detect_data['flag_2'] == False)
                          ].reset_index(drop=True)

        # 计算滚动平均值 先滑窗平均再进行后面的筛选 不会使数据间断 更有说服力
        # window_size = 5
        # df_dech = df_dech.rolling(window_size).mean()
        # df_dech = df_dech.dropna().reset_index(drop=True)

        # 限制 flow_instantaneous < 1000
        df_dech = df_dech[(df_dech['flow_instantaneous'] < 4000) & (df_dech['flow_instantaneous'] > 0)].reset_index(
            drop=True)
        # 限制 temp_chw_in < 30
        df_dech = df_dech[df_dech['temp_chw_in'] < 40].reset_index(drop=True)
        # 限制 temp_chw_out < 40
        df_dech = df_dech[df_dech['temp_chw_out'] < 50].reset_index(drop=True)
        # threshold = 2  # 设定一个通常的阈值（比如数据点超过2倍标准差范围之外会被视为异常值)
        # df_dech = df_dech[~(df_dech['temp_chw_out'] > threshold)].reset_index(drop=True)
        # 限制 power_active < 800
        df_dech = df_dech[(df_dech['power_active'] < 800) & (df_dech['power_active'] > 10)].reset_index(drop=True)

        # # 异常值处理一——模型预处理
        # from sklearn.ensemble import IsolationForest
        # clf = IsolationForest(random_state=0).fit(df_dech)
        # df_dech = df_dech[clf.predict(df_dech) == 1]
        # # 异常值处理二——聚类
        # from sklearn.cluster import DBSCAN
        # outlier_detection = DBSCAN(min_samples=2, eps=3)
        # clusters = outlier_detection.fit_predict(df_dech)
        # mask = clusters == -1
        # df_dech = df_dech[~mask]
        # 异常值处理三——IQR
        # Q1 = df_dech['temp_chw_out'].quantile(0.25)
        # Q3 = df_dech['temp_chw_out'].quantile(0.75)
        # IQR = Q3 - Q1
        # df_dech = df_dech[~((df_dech['temp_chw_out'] < (Q1 - 1.5 * IQR)) | (df_dech['temp_chw_out'] > (Q3 + 1.5 * IQR)))]

        # X = df_dech[["temp_chw_in", "temp_chw_out", "flow_instantaneous", "ratio_load", "ratio_load_verified", "ratio_load_diff"]].values
        X = df_dech[["temp_chw_in", "temp_chw_out", "flow_instantaneous", "ratio_load", "ratio_load_verified", "ratio_load_diff"]]
        Y = df_dech[["power_active"]].values

        return X, Y, df_dech


    def Power_Vs_Load_ratio_2D(self, start_time, end_time):
        """
        Parameters
        X :[tchin, tchout, Vch, tcdin]
        tchin :冷机冷冻水进水温度
        tchout :冷机冷冻水出水温度
        Vch :冷机冷冻水流量
        tcdin :冷机冷却水进水温度
        """
        sql_dech = '''
        select data_time, temp_chw_in, temp_chw_out, flow_instantaneous, power_active
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        '''.format(self.system_id, self.device_name, start_time, end_time)

        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)
        df_dech['CL'] = df_dech['flow_instantaneous'] * (
                    df_dech['temp_chw_in'] - df_dech['temp_chw_out']) * 4.1868 * 1000 / 3600
        df_dech['ratio_load_verified'] = df_dech['CL']/self.Cap

        detect_data = pd.DataFrame()
        detect_data.index = df_dech['data_time']
        detect_data['power_active'] = df_dech['power_active'].values
        detect_data['flag_1'] = self.level_shift_ad.fit_detect(detect_data['power_active'])
        detect_data['flag_2'] = self.quantile_ad.fit_detect(detect_data['power_active'])
        detect_data = detect_data.reset_index(drop=False)
        # 这里会自动对齐正确的index
        df_dech = df_dech[(detect_data['flag_1'] == False) &
                          (detect_data['flag_2'] == False)
                          ].reset_index(drop=True)

        # 计算滚动平均值 先滑窗平均再进行后面的筛选 不会使数据间断 更有说服力
        # window_size = 5
        # df_dech = df_dech.rolling(window_size).mean()
        # df_dech = df_dech.dropna().reset_index(drop=True)

        # 限制 flow_instantaneous < 1000
        df_dech = df_dech[(df_dech['flow_instantaneous'] < 4000) & (df_dech['flow_instantaneous'] > 0)].reset_index(
            drop=True)
        # 限制 temp_chw_in < 30
        df_dech = df_dech[df_dech['temp_chw_in'] < 40].reset_index(drop=True)
        # 限制 temp_chw_out < 40
        df_dech = df_dech[df_dech['temp_chw_out'] < 50].reset_index(drop=True)
        # threshold = 2  # 设定一个通常的阈值（比如数据点超过2倍标准差范围之外会被视为异常值)
        # df_dech = df_dech[~(df_dech['temp_chw_out'] > threshold)].reset_index(drop=True)
        # 限制 power_active < 800
        df_dech = df_dech[(df_dech['power_active'] < 800) & (df_dech['power_active'] > 10)].reset_index(drop=True)

        # # 异常值处理一——模型预处理
        # from sklearn.ensemble import IsolationForest
        # clf = IsolationForest(random_state=0).fit(df_dech)
        # df_dech = df_dech[clf.predict(df_dech) == 1]
        # # 异常值处理二——聚类
        # from sklearn.cluster import DBSCAN
        # outlier_detection = DBSCAN(min_samples=2, eps=3)
        # clusters = outlier_detection.fit_predict(df_dech)
        # mask = clusters == -1
        # df_dech = df_dech[~mask]
        # 异常值处理三——IQR
        # Q1 = df_dech['temp_chw_out'].quantile(0.25)
        # Q3 = df_dech['temp_chw_out'].quantile(0.75)
        # IQR = Q3 - Q1
        # df_dech = df_dech[~((df_dech['temp_chw_out'] < (Q1 - 1.5 * IQR)) | (df_dech['temp_chw_out'] > (Q3 + 1.5 * IQR)))]

        # X = df_dech[["temp_chw_in", "temp_chw_out", "flow_instantaneous", "ratio_load", "ratio_load_verified", "ratio_load_diff"]].values
        plt.scatter(df_dech["ratio_load_verified"], df_dech["power_active"])
        plt.title('Power_Vs_Load_ratio {} {}-{} '.format(self.device_name,start_time,end_time))
        # plt.legend()
        plt.show()

        return None


    def Power_Vs_Load_ratio_3D(self, start_time, end_time):
        """
        Parameters
        X :[tchin, tchout, Vch, tcdin]
        tchin :冷机冷冻水进水温度
        tchout :冷机冷冻水出水温度
        Vch :冷机冷冻水流量
        tcdin :冷机冷却水进水温度
        """
        sql_dech = '''
        select data_time, temp_chw_in, temp_chw_out, flow_instantaneous, p_condenser,p_evaporator, power_active
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        '''.format(self.system_id, self.device_name, start_time, end_time)

        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)
        df_dech['CL'] = df_dech['flow_instantaneous'] * (
                    df_dech['temp_chw_in'] - df_dech['temp_chw_out']) * 4.1868 * 1000 / 3600
        df_dech['ratio_load_verified'] = df_dech['CL']/self.Cap

        detect_data = pd.DataFrame()
        detect_data.index = df_dech['data_time']
        detect_data['power_active'] = df_dech['power_active'].values
        detect_data['flag_1'] = self.level_shift_ad.fit_detect(detect_data['power_active'])
        detect_data['flag_2'] = self.quantile_ad.fit_detect(detect_data['power_active'])
        detect_data = detect_data.reset_index(drop=False)
        # 这里会自动对齐正确的index
        df_dech = df_dech[(detect_data['flag_1'] == False) &
                          (detect_data['flag_2'] == False)
                          ].reset_index(drop=True)

        # 计算滚动平均值 先滑窗平均再进行后面的筛选 不会使数据间断 更有说服力
        # window_size = 5
        # df_dech = df_dech.rolling(window_size).mean()
        # df_dech = df_dech.dropna().reset_index(drop=True)

        # 限制 flow_instantaneous < 1000
        df_dech = df_dech[(df_dech['flow_instantaneous'] < 4000) & (df_dech['flow_instantaneous'] > 0)].reset_index(
            drop=True)
        # 限制 temp_chw_in < 30
        df_dech = df_dech[df_dech['temp_chw_in'] < 40].reset_index(drop=True)
        # 限制 temp_chw_out < 40
        df_dech = df_dech[df_dech['temp_chw_out'] < 50].reset_index(drop=True)
        # threshold = 2  # 设定一个通常的阈值（比如数据点超过2倍标准差范围之外会被视为异常值)
        # df_dech = df_dech[~(df_dech['temp_chw_out'] > threshold)].reset_index(drop=True)
        # 限制 power_active < 800
        df_dech = df_dech[(df_dech['power_active'] < 800) & (df_dech['power_active'] > 10)].reset_index(drop=True)

        # # 异常值处理一——模型预处理
        # from sklearn.ensemble import IsolationForest
        # clf = IsolationForest(random_state=0).fit(df_dech)
        # df_dech = df_dech[clf.predict(df_dech) == 1]
        # # 异常值处理二——聚类
        # from sklearn.cluster import DBSCAN
        # outlier_detection = DBSCAN(min_samples=2, eps=3)
        # clusters = outlier_detection.fit_predict(df_dech)
        # mask = clusters == -1
        # df_dech = df_dech[~mask]
        # 异常值处理三——IQR
        # Q1 = df_dech['temp_chw_out'].quantile(0.25)
        # Q3 = df_dech['temp_chw_out'].quantile(0.75)
        # IQR = Q3 - Q1
        # df_dech = df_dech[~((df_dech['temp_chw_out'] < (Q1 - 1.5 * IQR)) | (df_dech['temp_chw_out'] > (Q3 + 1.5 * IQR)))]

        # X = df_dech[["temp_chw_in", "temp_chw_out", "flow_instantaneous", "ratio_load", "ratio_load_verified", "ratio_load_diff"]].values
        # plt.scatter(df_dech["ratio_load_verified"], df_dech["power_active"])
        # plt.title('Power_Vs_Load_ratio {} {}-{} '.format(self.device_name,start_time,end_time))
        # plt.legend()

        # 绘制散点图
        fig = plt.figure()
        ax = Axes3D(fig)
        x = df_dech["ratio_load_verified"]
        y = df_dech["p_condenser"]
        z = df_dech["power_active"]

        ax.scatter(x, y, z)

        # 添加坐标轴(顺序是Z, Y, X)
        ax.set_zlabel('power_active', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('p_condenser', fontdict={'size': 15, 'color': 'blue'})
        ax.set_xlabel('ratio_load_verified', fontdict={'size': 15, 'color': 'red'})
        ax.set_title(str(self.device_name)+' con\n'+ start_time[5:10] + '\nto\n' + end_time[5:10], fontdict={'size': 18, 'color': 'blue'}, loc='left', y=0.5)
        plt.savefig(str(self.device_name)+' con'+ start_time[5:10] + 'to' + end_time[5:10]+'.png')
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        x = df_dech["ratio_load_verified"]
        y = df_dech["p_evaporator"]
        z = df_dech["power_active"]

        ax.scatter(x, y, z)

        # 添加坐标轴(顺序是Z, Y, X)
        ax.set_zlabel('power_active', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('p_evaporator', fontdict={'size': 15, 'color': 'blue'})
        ax.set_xlabel('ratio_load_verified', fontdict={'size': 15, 'color': 'red'})
        ax.set_title(str(self.device_name)+' eva\n'+ start_time[5:10] + '\nto\n' + end_time[5:10], fontdict={'size': 18, 'color': 'blue'}, loc='left', y=0.5)
        plt.savefig(str(self.device_name)+' eva'+ start_time[5:10] + 'to' + end_time[5:10]+'.png')
        plt.show()

        scatter3d = Scatter3D()
        scatter3d.set_global_opts(
            title_opts=opts.TitleOpts(title=str(self.device_name) + ' eva\n' + start_time + '\n' + end_time + '\n' +
                                            'x:ratio_load_verified\ny:p_condenser\nz:power_active'),
            visualmap_opts=opts.VisualMapOpts(range_color=["#00BFFF", "#FF4500"]),
        )

        data = df_dech[["ratio_load_verified", "p_condenser", "power_active"]].values.tolist()
        scatter3d.add("", data)

        scatter3d.set_series_opts(
            xaxis3d_opts=opts.AxisOpts(type_="value"),
            yaxis3d_opts=opts.AxisOpts(type_="value"),
            zaxis3d_opts=opts.AxisOpts(type_="value"),
        )


        scatter3d.render("3d_scatter_{}.html".format(self.device_name))

        return None



    #####################################Power结束######################################################

if __name__ == "__main__":
    for i in ['dech01', 'dech02', 'dech03']:
    # for i in ['dech02']:
        a = Chiller("3051", i, 1100, 1775, 5, 535.6)
        # b = a.power_get_data(start_time='2023-07-11 00:00:00', end_time='2023-07-12 00:00:00')[0]
        # acl = a.power_get_data(start_time='2023-07-11 00:00:00', end_time='2023-07-12 00:00:00')[1]
        # pre = a.power_predict(b)
        # simu = a.power_predict(pd.read_csv('power_data_simulated7.11-7.12.csv').values)
        # plt.plot(pre, label='pre')
        # plt.plot(simu, label='simu')
        # plt.legend()
        # plt.title('7.11-7.12')
        # plt.show()
        # print(type(b))
        R = a.ratio_load_verify('2023-07-10 00:00:00', '2023-07-12 09:59:59')[2]
        print(R)
        # a.Power_Vs_Load_ratio_2D('2023-07-01 00:00:00', '2023-07-12 09:59:59')
        a.Power_Vs_Load_ratio_3D('2023-03-01 00:00:00', '2023-07-12 09:59:59')
        # print('ratio_load的误差从1%-10%，∴自己算吧')