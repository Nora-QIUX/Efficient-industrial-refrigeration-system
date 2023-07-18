import datetime
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

        # self.n = n

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
        np.save('model_new/{}_{}_Max_Q_params'.format(self.system_id, self.device_name), popt)

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
            'model_new/{}_{}_Max_Q_params.npy'.format(
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
        # df_dech['ratio_load_verified'] = df_dech['CL'] / self.Cap

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

        # X = df_dech[["CL", "p_condenser", "p_evaporator"]].values
        X = df_dech[["CL", "p_condenser"]].values
        Y = df_dech[["power_active"]].values

        return X, Y, df_dech

    def power_fun(self, X, *params):
        """
        用于拟合功率模型的函数
        Parameters
        ----------
        X :[ratio_load_verified, p_condenser]
        ratio_load_verified :负荷率
        p_condenser :冷凝压力
        a0, a1, b0:功率模型系数

        Returns
        W_Ch ：冷机功率函数
        -------
        """
        cl = X[0]
        cond = X[1]
        # evap = X[2]
        # a0, a1, b0, a2 = params
        a0, a1, b0= params
        # if self.device_name == 'dech03':
        #     W_Ch = a0 * ratio + a1 * pow(cond, 3.8)
        # else:
        #     W_Ch = a0 * ratio + a1 * cond
        # W_Ch = a0 * cl + a1 * pow(cond, b0) + a2 * evap
        W_Ch = a0 * cl + a1 * pow(cond, b0)
        return W_Ch

    def power_train(self, start_time, end_time):
        X, Y, Z = self.power_get_data(start_time, end_time)
        X1 = X[:,0]
        X2 = X[:,1]
        # X3 = X[:,2]
        # X = (X1,X2,X3)
        X = (X1,X2)
        Y = Y.reshape(1, -1)[0]
        func = self.power_fun
        # p0 = [0.0, 0.0, 0.0, 0.0]
        p0 = [0.0, 0.0, 0.0]
        popt, pcov = curve_fit(func, X, Y, p0=p0, maxfev=50000)
        np.save('model_new/{}_{}_power_params'.format(self.system_id, self.device_name), popt)

    def power_predict(self, X):
        params = np.load('model_new/{}_{}_power_params.npy'.format(
                self.system_id,
                self.device_name
                ))

        X1 = X[:,0]
        X2 = X[:,1]
        # X3 = X[:,2]
        # X = (X1, X2, X3)
        X = (X1, X2)
        # a0, a1, b0, a2 = params
        a0, a1, b0= params
        # W_Ch = np.array(self.power_fun(X, a0, a1, b0, a2))
        W_Ch = np.array(self.power_fun(X, a0, a1, b0))
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
    for i in ['dech01', 'dech02', 'dech03']:
    # for i in ['dech02']:
        a = Chiller("3051", i, 1100, 1775, 5, 535.6)
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        yesterday = datetime.datetime.now().replace(hour=23, minute=59, second=59) - datetime.timedelta(days=1)
        today = yesterday + datetime.timedelta(seconds=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d %H:%M:%S')
        today_str = today.strftime('%Y-%m-%d %H:%M:%S')
        a.power_train(start_time='2023-03-01 00:00:00', end_time=yesterday_str)
        a.Max_Q_train(start_time='2023-03-01 00:00:00', end_time=yesterday_str)
        P = a.power_predict(a.power_get_data(today_str, now_str)[0])
        Y = a.power_get_data(today_str, now_str)[1]
        try:
            mape = mape_sklearn(Y, P)
            print('{}的new_mape是{} 预测时段是{}~~{} 训练时段到{}为止'.format(i, str(mape), today_str, now_str, yesterday_str))
        except Exception as e:
            print('计算{}的mape时发生异常:{}'.format(i, str(e)))

    print('\ndech01的new_mape是0.0050829412042943225 预测时段是2023-07-14 00:00:00~~2023-07-14 09:58:40 训练时段到2023-07-13 23:59:59为止\n'
          '计算dech02的mape时发生异常:Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.\n'
          'dech03的new_mape是0.05832216531240657 预测时段是2023-07-14 00:00:00~~2023-07-14 09:58:45 训练时段到2023-07-13 23:59:59为止')
    print('\n')
    print('dech01的new_mape是0.0456526041787258')
    print('dech02的new_mape是0.07151840236222964')
    print('dech03的new_mape是0.03922976287771474')
    print('\n')
    print('交大：dech01的mape是0.04025351343624178')
    print('交大：dech02的mape是0.8320075196251075')
    print('交大：dech03的mape是0.1016757914874486')
    print('\n')
    print('cond^3：dech03的mape是0.052023438891491684')
    print('cond^4：dech03的mape是0.04025850786055403')
    print('cond^5：dech03的mape是0.05164309485298308')
    print('cond^4.5：dech03的mape是0.04556436017839312')
    print('\n')

    # current_dir = os.getcwd()
    # file_path = os.path.join(current_dir, "dech03_new_mape_df.csv")
    # if os.path.isfile(file_path):
    #     dech03_new_mape_df = pd.read_csv(file_path)
    # else:
    #     dech03_new_mape_df = pd.DataFrame(columns=['cond_pow_level','mape'])
    #     for pow_n in np.arange(3, 5.1, 0.1):
    #         a = Chiller("3051", 'dech03', 1100, 1775, 5, 535.6, pow_n)
    #         a.power_train(start_time='2023-03-01 00:00:00', end_time='2023-07-09 23:59:59')
    #         P = a.power_predict(a.power_get_data('2023-07-10 00:00:00', '2023-07-12 09:59:59')[0])
    #         Y = a.power_get_data('2023-07-10 00:00:00', '2023-07-12 09:59:59')[1]
    #         mape = mape_sklearn(Y, P)
    #         new_row = {'cond_pow_level': pow_n, 'mape': mape}
    #         dech03_new_mape_df = dech03_new_mape_df.append(new_row, ignore_index=True)
    #         print('dech03,cond^{}：mape={}'.format(pow_n, str(mape)))
    # plt.plot(dech03_new_mape_df['cond_pow_level'], dech03_new_mape_df['mape'])
    # plt.title('pow_n VS mape')
    # plt.savefig('pow_n VS mape.png')
    # plt.show()
    # print(dech03_new_mape_df)
    print('cond^3.8：dech03的mape是0.03923 最小')
    print('\n')