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

    # tofit=func  fun=predict  fit=train
    #####################################Max_Q开始#######################################################
    # 没用到，说是没有厂家提供的数值，直接设定的c0为1.0计算的，先预留着吧。按理说，当前工况下的最大制冷量需要跟随当前电功率改变

    def Max_Q_get_data(self, start_time, end_time):
        """
        冷机最大制冷量模型拟合方法
        Parameters
        ----------
        X :最大制冷量模型输入变量[tchin, tcdin]
        tchin :冷机冷冻水进水温度
        tcdin :冷机冷却水进水温度

        Y :最大制冷量Max_Q
        -------
        """
        sql_dech = '''
        select data_time, flow_instantaneous, temp_chw_in, temp_chw_out, temp_cow_in, power_active
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        '''.format(self.system_id, self.device_name, start_time, end_time)
        # "3051", "dech01", "2023-02-01 00:00:00", "2023-05-09 23:59:59"
        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)
        df_dech['CL'] = df_dech['flow_instantaneous'] * (
                    df_dech['temp_chw_in'] - df_dech['temp_chw_out']) * 4.1868 * 1000 / 3600


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

        # 限制 flow_instantaneous < 1000
        df_dech= df_dech[(df_dech['flow_instantaneous'] < 4000) & (df_dech['flow_instantaneous'] > 0)].reset_index(drop=True)
        # 限制 temp_chw_in < 30
        df_dech = df_dech[df_dech['temp_chw_in'] < 40].reset_index(drop=True)
        # 限制 power_active < 800
        df_dech = df_dech[(df_dech['power_active'] < 800) & (df_dech['power_active'] > 10)].reset_index(drop=True)
        # 限制 CL > 0
        df_dech = df_dech[df_dech['CL'] > 0].reset_index(drop=True)

        X = df_dech[["temp_chw_in","temp_cow_in"]].values
        Y = df_dech[["CL"]].values

        return X, Y, df_dech

    def Max_Q_fun(self, X, *params):
        """
        用于拟合最大制冷量模型的函数
        Parameters
        ----------
        X :[tchin, tcdin]
        tchin :冷机冷冻水进水温度
        tcdin :冷机冷却水进水温度
        params:[c0, c1, c2, c3, c4, c5]最大制冷量模型系数
        -------
        """
        tchin = X[0]
        tcdin = X[1]
        c0, c1, c2, c3, c4, c5 = params
        Q_Cap = self.Cap
        Max_Q = Q_Cap * (c0 + c1 * tchin + c2 * pow(tchin, 2) + c3 * tcdin + c4 * pow(tcdin, 2) + c5 * tchin * tcdin)
        return Max_Q

    def Max_Q_train(self, start_time, end_time):
        """
        冷机最大制冷量模型拟合方法
        Parameters
        ----------
        X :最大制冷量模型输入变量[tchin, tcdin]
        Y :最大制冷量Max_Q
        -------
        """
        X, Y, Z = self.Max_Q_get_data(start_time, end_time)
        X1 = X[:,0]
        X2 = X[:,1]
        X = (X1,X2)
        Y = Y.reshape(1, -1)[0]
        func = self.Max_Q_fun
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        popt, pcov = curve_fit(func, X, Y, p0=p0, maxfev=50000)
        np.save('model/{}_{}_Max_Q_params'.format(self.system_id, self.device_name), popt)

    def Max_Q_predict(self, X):
        """
        冷机最大制冷量模型
        Parameters
        ----------
        X :[tchin, tcdin]
        tchin :冷机冷冻水进水温度
        tcdin :冷机冷却水进水温度

        Returns
        Max_Q ：冷机最大制冷量
        -------
        """
        params = np.load(
            'model/{}_{}_Max_Q_params.npy'.format(
                self.system_id,
                self.device_name
                ))

        X1 = X[:,0]
        X2 = X[:,1]
        Q_Cap = self.Cap
        # c0, c1, c2, c3, c4, c5 = params
        c0, c1, c2, c3, c4, c5 = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        Max_Q = Q_Cap * (c0 + c1 * X1 + c2 * pow(X1, 2) + c3 * X2 + c4 * pow(X2, 2) + c5 * X1 * X2)
        return Max_Q

    #####################################Max_Q结束########################################################

    #####################################Power开始########################################################
    # Question:和薛博士的power计算逻辑不一样？有什么选择的说法嘛？
    # Answer:因为两者主要目的不一样，但后期会整合这两个出来一个更好的。
    def power_get_data(self, start_time, end_time):
        """
        Parameters
        X :[tchin, tchout, Vch, tcdin]
        tchin :冷机冷冻水进水温度
        tchout :冷机冷冻水出水温度
        Vch :冷机冷冻水流量
        tcdin :冷机冷却水进水温度
        """
        sql_dech = '''
        select data_time, temp_chw_in, temp_chw_out, flow_instantaneous, temp_cow_in, power_active
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        '''.format(self.system_id, self.device_name, start_time, end_time)

        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)

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

        X = df_dech[["temp_chw_in", "temp_chw_out", "flow_instantaneous", "temp_cow_in"]].values
        Y = df_dech[["power_active"]].values

        return X, Y, df_dech

    def power_get_data_by_label(self, start_time, end_time, label_temp: float):
        """
        Parameters
        X :[tchin, tchout, Vch, tcdin]
        tchin :冷机冷冻水进水温度
        tchout :冷机冷冻水出水温度
        Vch :冷机冷冻水流量
        tcdin :冷机冷却水进水温度
        """
        sql_dech = '''
        select data_time, temp_chw_in, temp_chw_out, flow_instantaneous, temp_cow_in, power_active, temp_set_chw_out
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        and temp_set_chw_out = {}
        '''.format(self.system_id, self.device_name, start_time, end_time, label_temp)

        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)

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

        X = df_dech[["temp_chw_in", "temp_chw_out", "flow_instantaneous", "temp_cow_in"]].values
        Y = df_dech[["power_active"]].values

        return X, Y, df_dech

    def power_fun(self, X, *params):
        """
        用于拟合功率模型的函数
        Parameters
        ----------
        X :[tchin, tchout, Vch, tcdin]
        tchin :冷机冷冻水进水温度
        tchout :冷机冷冻水出水温度
        Vch :冷机冷冻水流量
        tcdin :冷机冷却水进水温度
        a0, a1, a2, b0, b1, b2, b3, b4, b5 :功率模型系数

        Returns
        W_Ch ：冷机功率函数
        -------
        """
        tchin = X[0]
        tchout = X[1]
        Vch = X[2]
        tcdin = X[3]
        a0, a1, a2, b0, b1, b2, b3, b4, b5 = params
        COP_nom = self.Cop_ref
        Delta_tch = tchin - tchout
        PLR = (Delta_tch * Vch) / (self.Delta_tch_des * self.Vch_des)
        PLR_rev = a0 + a1 * PLR + a2 * pow(PLR, 2)
        Temp_rev = b0 + b1 * tchout + b2 * pow(tchout, 2) + b3 * tcdin + b4 * pow(tcdin, 2) + b5 * tchout * tcdin
        W_Ch = self.Cap * PLR_rev * Temp_rev / COP_nom  # 厂家提供数据，最大制冷量模型确定后，self.Cap换为Max_Q
        # 等换成模型时，需要添加输入参数Max_Q（array）【待定】，应用时用Max_Q_predict()的结果作为入参
        return W_Ch

    def power_train(self, start_time, end_time):
        X, Y, Z = self.power_get_data(start_time, end_time)
        X1 = X[:,0]
        X2 = X[:,1]
        X3 = X[:,2]
        X4 = X[:,3]
        X = (X1,X2,X3,X4)
        Y = Y.reshape(1, -1)[0]
        func = self.power_fun
        p0 = [0.0, 0.0, 0.0, 0.493, 0.0, 0.0, 0.0, 0.0, 0.0]
        popt, pcov = curve_fit(func, X, Y, p0=p0, maxfev=50000)
        np.save('model/{}_{}_power_params'.format(self.system_id, self.device_name), popt)

    def power_predict(self, X):
        params = np.load('model/{}_{}_power_params.npy'.format(
                self.system_id,
                self.device_name
                ))

        X1 = X[:,0]
        X2 = X[:,1]
        X3 = X[:,2]
        X4 = X[:,3]
        X = (X1, X2, X3, X4)
        a0, a1, a2, b0, b1, b2, b3, b4, b5 = params
        W_Ch = np.array(self.power_fun(X, a0, a1, a2, b0, b1, b2, b3, b4, b5))
        return W_Ch

    #####################################Power结束######################################################

    #####################################计算 Min_set_out_temp 开始######################################
    # 预测这个“出水温度设定值”是为了看当下工况决定的Max_Q是否可以做到让出水温度符合这个值。
    # 代码逻辑：“预测最低出水温度”+限制机器允许的最低温 得到所谓的出水温度设定值 temp_set_chw_out
    # e.g.当下工况有对应的Max_Q 据此结合流量和冷冻水进水温度 出水温度最低只能是4℃ 如果你设定出水温度为4℃以下 那肯定不行
    def Min_set_out_temp_get_data(self, start_time, end_time):
        """
        Parameters
        X :[tchin, Vch]
        tchin :冷机冷冻水进水温度
        Vch :冷机冷冻水流量
        """
        sql_dech = '''
        select data_time, temp_chw_in, temp_chw_out, flow_instantaneous, temp_cow_in, power_active, temp_set_chw_out
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        '''.format(self.system_id, self.device_name, start_time, end_time)

        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)

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

        X = df_dech[["temp_chw_in", "flow_instantaneous"]].values
        Y = df_dech[["temp_set_chw_out"]].values

        return X,Y

    def Min_set_out_temp_calculate(self, X, Max_Q, tchout_set_LB):
        """
        获得冷机最小出水温度设定值
        Parameters
        ----------
        X :[tchin, Vch]
        tchin :冷机冷冻水进水温度
        Vch :冷机冷冻水流量
        Max_Q :冷机最大制冷量  #后期 Max_Q 用 Max_Q_predict()的结果带入 [Nora]
        tchout_set_LL ：厂家给定的出水温度设定值下限 #Question：后期厂家提供是吗 Answer:对

        Returns
        tchout_set_min ：冷机最小出水温度设定值
        -------
        """
        tchin = X[:,0]
        Vch = X[:,1] / 3600 * self.pw  # m3/h to kg/s
        # print("---tchin---")
        # print(tchin) # 11.xx
        # print("---Vch * self.cpw---")
        # print(Vch * self.cpw) # 370-390
        # Question：我认为此函数目标是要得到一个数组 因为使用了Max_Q,这个值应该根据不同的水温有所改变，这个最小设定温度应该也是
        # Answer:对
        tchout_set_min = tchin - Max_Q / (Vch * self.cpw) #0.67-1.61
        # Max_Q取机器上限时 此值为0.6721213052727943 数值很小 但后期随着Max_Q改变 会变大 就可以和厂家设定的最低温度值比较了
        # print(tchout_set_min.min(),tchout_set_min.max())
        tchout_set_min = np.where(tchout_set_min < tchout_set_LB, tchout_set_LB, tchout_set_min)

        return tchout_set_min
    #####################################计算 Min_set_out_temp 结束######################################

    #####################################计算 冷机制冷量 开始##############################################
    # 单纯计算 当下工况需要的制冷量
    def Q_get_data(self, start_time, end_time):
        """
        Parameters
        X :[tchin, tchout, Vch]
        tchin :冷机冷冻水进水温度
        tchout :冷机的冷冻水出水温度
        Vch :冷机冷冻水流量
        """
        sql_dech = '''
        select data_time, temp_chw_in, temp_chw_out, flow_instantaneous, temp_cow_in, power_active
        from dech_{}_l1
        where device_name = '{}'
        and data_time between '{}' and '{}'
        '''.format(self.system_id, self.device_name, start_time, end_time)

        conn = conn_000012
        df_dech = pd.read_sql(sql_dech, conn)

        df_dech = df_dech.dropna().reset_index(drop=True)

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

        X = df_dech[["temp_chw_in", "temp_chw_out", "flow_instantaneous"]].values

        return X

    def Q_calculate(self, X):
        """
        获得冷机制冷量
        Parameters
        ----------
        X :[tchin, tchout, Vch]
        tchin :冷机冷冻水进水温度
        tchout :冷机的冷冻水出水温度
        Vch :冷机冷冻水流量

        Returns
        Q ：冷机的制冷量
        -------
        """
        tchin = X[:,0]
        tchout = X[:,1]
        Vch = X[:,2] / 3600 * self.pw  # m3/h to kg/s
        Q = Vch * self.cpw * (tchin - tchout)

        return Q

    #####################################计算 冷机制冷量 结束##############################################

    #####################################power_cond开始#################################################
    # Question: mape维持在0.02以下 0.0192之类的 有分段的必要吗？
    # Answer: 当初为厦门项目考虑，此项目不需要，因为分段会造成不连续，寻优时会有奇怪的点。
    # Question: tchout_set_list, tcdin_seg_list没有什么特殊的讲究吧 如果是，那均分一下区间即可
    # Answer: 没有 均分即可 分温度区间训练电量模型 单纯为了提高精度
    def get_train_data(self, tchout_set_list, tcdin_seg_list, arr_train_data: np.array):
        X_list = []
        Y_list = []
        for i in range(len(tchout_set_list) - 1):
            X_i = []
            Y_i = []
            #选出 arr_train_data 中第二列（即索引为 1 的列）中数值在 tchout_set_list[i] 和 tchout_set_list[i+1] 之间的所有行
            arr_train_data_i = arr_train_data[np.where((arr_train_data[:, 1] >= tchout_set_list[i])
                                                       & (arr_train_data[:, 1] < tchout_set_list[i + 1]))]

            for j in range(len(tcdin_seg_list) - 1):
                # 选出 arr_train_data 中第四列（即索引为 3 的列）中数值在 tchout_seg_list[i] 和 tchout_seg_list[i+1] 之间的所有行
                arr_train_data_i_j = arr_train_data_i[np.where((arr_train_data_i[:, 3] >= tcdin_seg_list[j])
                                                               & (arr_train_data_i[:, 3] < tcdin_seg_list[j + 1]))]
                # 转置 =.T
                arr_train_data_i_j = arr_train_data_i_j.transpose()
                # 把arr_train_data_i_j的最后一行作为Y_i_j
                X_i_j = arr_train_data_i_j[0:len(arr_train_data_i_j) - 1]
                Y_i_j = arr_train_data_i_j[-1]
                X_i.append(X_i_j)
                Y_i.append(Y_i_j)

            X_list.append(X_i)
            Y_list.append(Y_i)

        return X_list, Y_list

    def Power_cond_tofit(self, tchout_set_list, tcdin_list, X_list, Y_list):
        """
        拟合功率模型（根据冷冻水出水温度和冷却水出水温度分段）
        :param tchout_set_list:
        :param tcdin_list:
        :param arr_train_data: 训练数据[tchin, tchout, Vch, tcdin, W_Ch]
        :return:
        """
        #更新系数
        pow_coef = []
        for i in range(len(tchout_set_list) - 1):
            pow_coef_i = []
            for j in range(len(tcdin_list) - 1):
                X_i_j = X_list[i][j]
                Y_i_j = Y_list[i][j]
                # 判断训练数据量是否满足最小数据量
                train_data_min = 50
                if len(X_i_j[0]) > train_data_min:
                    pow_coef_i_j_values = list(self.power_train(X_i_j, Y_i_j))
                else:
                    pow_coef_i_j_values = None
                pow_coef_i.append(pow_coef_i_j_values)
            pow_coef.append(pow_coef_i)

        #存系数
        index_list = []
        for i in range(len(tchout_set_list) - 1):
            index_i = "power_coef" + "_" + str(tchout_set_list[i]) + "_" + str(tchout_set_list[i + 1])
            index_list.append(index_i)

        columns_list = []
        for j in range(len(tcdin_list) - 1):
            columns_j = "tchin" + "_" + str(tcdin_list[j]) + "_" + str(tcdin_list[j + 1])
            columns_list.append(columns_j)

        pow_coef_df = pd.DataFrame(pow_coef, index=index_list, columns=columns_list)
        # 功率模型系数为空的片程，用邻近温度片程的功率模型系数替代
        pow_coef_df = pow_coef_df.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)
        pow_coef_df = pow_coef_df.fillna(method="pad", axis=1).fillna(method="backfill", axis=1)

        # 转为json格式
        pow_coef_json = json.dumps(pow_coef_df.T.to_dict(orient="list"))

        return pow_coef_json

    def update_pow_coef(self, X_list, Y_list, Power_cond_coef, Power_cond_coef_old):
        """
        更新冷机功率模型系数方法
        :param X_list: 冷机功率模型辨识数据（输入变量）
        :param Y_list: 冷机功率模型辨识数据（输出变量）
        :param Power_cond_coef:新数据拟合的冷机模型系数（多工况）
        :param Power_cond_coef_old:冷机模型的旧系数（多工况）
        :return new_Power_cond_coef_list: 更新后的冷机模型系数
        """
        # 更新冷机不同出水温度功率模型系数
        # 取出来系数
        new_Power_cond_coef_list = []
        min_data_set = 50
        for i in range(len(Power_cond_coef)):
            X_list_i = X_list[i]
            Y_list_i = Y_list[i]
            new_Power_cond_coef_i = []
            for j in range(len(X_list_i)):
                X_list_i_j = X_list_i[j]
                Y_list_i_j = Y_list_i[j]
                Power_cond_coef_i_j = Power_cond_coef[i][j]
                Power_cond_coef_old_i_j = Power_cond_coef_old[i][j]

                #对比
                if X_list_i_j.size > min_data_set:
                    # 利用更新前模型系数预测功率
                    Y_list_prd_temp = self.power_train(X_list_i_j, Power_cond_coef_old_i_j)
                    # 计算更新前系数预测结果的RMSE
                    RMSE_CH_temp = mean_squared_error(Y_list_i_j, Y_list_prd_temp)
                    # 利用新模型系数预测功率
                    Y_list_prd_new_temp = self.power_train(X_list_i_j, Power_cond_coef_i_j)
                    # 计算新模型系数预测结果的RMSE
                    RMSE_CH_new_temp = mean_squared_error(Y_list_i_j, Y_list_prd_new_temp)

                    if RMSE_CH_new_temp < RMSE_CH_temp:
                        new_Power_cond_coef_i_j = Power_cond_coef_i_j
                    else:
                        new_Power_cond_coef_i_j = Power_cond_coef_old_i_j
                else:
                    new_Power_cond_coef_i_j = Power_cond_coef_old_i_j

                new_Power_cond_coef_i.append(new_Power_cond_coef_i_j)

            new_Power_cond_coef_list.append(new_Power_cond_coef_i)

        return new_Power_cond_coef_list

    def Power_cond_fun(self, X, Power_cond_coef):
        """
        冷机功率分段模型（以冷冻水出水温度分段）
        Parameters
        ----------
        X :[tchin, tchout, Vch, tcdin]
        tchin :冷机冷冻水进水温度
        tchout :冷机冷冻水出水温度
        Vch :冷机冷冻水流量
        tcdin :冷机冷却水进水温度
        Power_cond_coef :不同段的功率模型系数[Power_coef_1, Power_coef_2,..., Power_coef_N]，
                         其中，Power_coef_i = [Power_coef_i_1,Power_coef_i_2,...,Power_coef_i_m]

        Returns
        W_Ch ：冷机功率
        -------
        """
        tchout = X[1]
        tcdin = X[3]
        # tchout_seg_list = list(np.linspace(5, 12, 8))
        # tchout_seg_list = [4.75, 5.75, 6.75, 7.75, 8.75, 9.75, 10.75, 11.75, 12.75]
        tchout_seg_list = [5, 12]

        tcdin_seg_list = [0, 100]

        # 为当前样本择定系数1
        if tchout < tchout_seg_list[0]:
            # 低于下限
            Power_coef_i = Power_cond_coef[0]
        elif tchout > tchout_seg_list[-1]:
            #高于上限
            Power_coef_i = Power_cond_coef[-1]
        else:
            # 寻找符合区间
            for i in range(len(tchout_seg_list) - 1):
                if (tchout_seg_list[i] <= tchout <= tchout_seg_list[i + 1]):
                    Power_coef_i = Power_cond_coef[i]

        # 为当前样本择定系数2
        if tcdin < tcdin_seg_list[0]:
            Power_coef_i_j = Power_coef_i[0]
        elif tcdin > tcdin_seg_list[-1]:
            Power_coef_i_j = Power_coef_i[-1]
        else:
            for j in range(len(tcdin_seg_list) - 1):
                if (tcdin_seg_list[j] <= tcdin <= tcdin_seg_list[j + 1]):
                    Power_coef_i_j = Power_coef_i[j]

        Wch = self.power_train(X, Power_coef_i_j)

        return Wch

    #####################################power_cond结束#################################################

if __name__ == "__main__":
    dech01 = Chiller("3051", 'dech01', 1100, 1775, 5, 535.6)
    dech02 = Chiller("3051", 'dech02', 1100, 1775, 5, 535.6)
    dech03 = Chiller("3051", 'dech03', 1100, 1775, 5, 535.6)
    label_temp_df = pd.read_sql(
        "select device_name,temp_set_chw_out,count(*) "
        "from dech_3051_l1 "
        "where data_time > '2023-07-01 00:00:00' "
        "group by device_name,temp_set_chw_out", conn_000012)
    label_temp_df = label_temp_df[label_temp_df['count'] > 12]
    for index, row in label_temp_df.iterrows():
        device_name = row['device_name']
        label_temp = float(row['temp_set_chw_out'])
        device_instance = globals()[device_name]
        X = device_instance.power_get_data_by_label(start_time='2023-07-01 00:00:00', end_time='2023-07-12 00:00:00', label_temp=label_temp)[0]
        origin_pre = device_instance.power_predict(X)
        X_plus2 = X.copy()
        X_plus2[:, 1] += 2
        plus2_pre = device_instance.power_predict(X_plus2)
        X_minus2 = X.copy()
        X_minus2[:, 1] -= 2
        minus2_pre = device_instance.power_predict(X_minus2)
        plt.plot(origin_pre, label='origin_pre')
        plt.plot(plus2_pre, label='plus2_pre')
        plt.plot(minus2_pre, label='minus2_pre')
        plt.legend()
        plt.title('7.1-7.12 {}_{}'.format(device_name, label_temp))
        plt.show()
