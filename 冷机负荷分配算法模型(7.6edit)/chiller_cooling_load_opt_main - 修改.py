import pandas as pd
import time
import psycopg2
import numpy as np
from scipy.optimize import curve_fit
from sko.PSO import PSO
import json
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings("ignore")

"""
2023.5.27 添加了冷冻水供水总管温度设定值调整方法
"""

# 设置温度
# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import sys
from ALi_Class import Local_2_AliCloud
from chiller_class import Chiller
from tools import conn_000012


# PLC的控制类
PLC_controler = Local_2_AliCloud()

"""
负荷优化算法类
"""
class Chillers_CL_allocate_system(object):
    ESR_LL = 0.5  # 单位%
    ESR_UL = 10  # 单位%

    def __init__(self, total_Chillers_list: list):
        self.total_chiller_num = len(total_Chillers_list)
        self.total_Chiller_list = total_Chillers_list
        # 获取所有冷机的列表
        self.total_Chiller_code_list = []
        self.total_Chiller_cap_list = []
        for chiller in total_Chillers_list:
            self.total_Chiller_code_list.append(chiller.device_name)
            self.total_Chiller_cap_list.append(chiller.Cap)

    def _check_action(self, tchout_list, chiller_list):
        """
        检查负荷分配策略和冷机列表是否能对应得上
        Parameters
        ----------
        tchout_list :各台运行冷机冷冻水进水温度[tchin_1,tchin_2,...,tchin_N]
        chiller_list :各台运行冷机的类[CH_1,CH_2,...,CH_N]
        -------

        """
        NA = len(tchout_list)
        NC = len(chiller_list)
        if NA == NC:
            return True
        else:
            raise Exception("chiller nums not equal action")

    def subSYS_Power_func(self, tchin_list, tchout_list, Vch_list, tcdin_list, Power_coef_list, chiller_list):
        """
        计算冷水机组子系统总功率
        Parameters
        ----------
        tchin_list :各台运行冷机冷冻水进水温度[tchin_1,tchin_2,...,tchin_N]
        tchout_list :各台运行冷机冷冻水出水温度[tchout_1,tchout_2,...,tchout_N]
        Vch_list :各台运行冷机的冷冻水流量[Vch_1.Vch_2,...,Vch_N]
        tcdin_list :各台运行冷机的冷却水进水温度[tcdin_1,tcdin_2,...,tcdin_N]
        Power_coef_list :各台运行冷机的功率模型系数[Power_coef_1,Power_coef_2,...,Power_coef_N]
        chiller_list :各台运行冷机的类[CH_1,CH_2,...,CH_N]

        Returns
        Power :冷机子系统的功率
        -------

        """
        Power = 0
        Num = len(chiller_list)
        if not (self._check_action(tchout_list, chiller_list)):
            return None
        for i in range(Num):
            Chiller_i = chiller_list[i]
            Power_coef_i = Power_coef_list[i]
            tchin_i = tchin_list[i]
            tchout_i = tchout_list[i]
            Vch_i = Vch_list[i]
            tcdin_i = tcdin_list[i]
            input_para = [tchin_i, tchout_i, Vch_i, tcdin_i]
            Power_i = Chiller_i.Power_cond_fun(input_para, Power_coef_i)
            Power = Power + Power_i

        return Power

    def _get_opt_func(self, tchin_list, Vch_list, tcdin_list, Power_coef_list, chiller_list):
        """
        获取优化目标函数
        Parameters
        ----------
        tchin_list :各台运行冷机冷冻水进水温度[tchin_1,tchin_2,...,tchin_N]
        Vch_list :各台运行冷机冷冻水流量[Vch_1.Vch_2,...,Vch_N]
        tcdin_list :各台运行冷机冷却水进水温度[tcdin_1,tcdin_2,...,tcdin_N]
        power_coef_list :各台运行冷机的功率模型系数[Power_coef_1,Power_coef_2,...,Power_coef_N]
        chiller_list :各台运行冷机的类[CH_1,CH_2,...,CH_N]

        Returns
        _opt_func ：冷机子系统的功率函数（冷机子系统功率与出口水温设定值的函数关系）
        -------

        """

        def _opt_func(X: list):
            """
            Parameters
            ----------
            X :各台运行冷机冷冻水出水温度的优化设定值[tchout_set_opt_1,tchout_set_opt_2,...,tchout_set_opt_N]

            Returns
            ESR ：节能率
            -------

            """
            power_sum_opt = self.subSYS_Power_func(tchin_list, X, Vch_list, tcdin_list, Power_coef_list, chiller_list)

            return power_sum_opt

        return _opt_func

    def cons_tsup_set(self, chiller_list, Vch_list, tsup_set, tchout_list, tchout_opt_list):
        """
        获取冷冻水总管供水温度设定值的约束函数
        Parameters
        ----------
        chiller_list :各台运行冷机的类[CH_1,CH_2,...,CH_N]
        tsup_set :供水温度设定值
        tchout_list: 当前各台运行冷机的出水温度
        tchout_opt_list：优化后的各台冷机的出水温度

        Returns
        cons_tuple ：约束函数元组
        -------
        """
        # 所有冷机的冷冻侧总流量
        # 约束温度 加权平均的演化[Nora]
        Vch_total = sum(Vch_list)
        temp = Vch_total * tsup_set
        temp2 = 0
        for i in range(len(chiller_list)):
            tchout_opt_i = tchout_opt_list[i]
            Vch_i = Vch_list[i]
            temp2 += Vch_i * tchout_opt_i

        delta = temp2 - temp

        return delta

    def _get_cons_fuc(self, chiller_list, CL_total, tchin_list, Vch_list, tcdin_list, Max_Q_coef_list, tchout_list,
                      tsup_set):  # tchout 是输入
        """
        获取冷机的约束函数
        Parameters
        ----------
        chiller_list :各台运行冷机的类[CH_1,CH_2,...,CH_N]
        CL_total :末端负荷
        tchin_list :各台运行冷机的冷冻水进水温度[tchin_1,tchin_2,...,tchin_N]
        Vch_list :各台运行冷机的冷冻水流量[Vch_1.Vch_2,...,Vch_N]
        tcdin_list :各台运行冷机的冷却水进水温度[tcdin_1.tcdin_2,...,tcdin_N]
        Max_Q_coef_list :各台运行冷机的最大制冷量模型系数[Max_Q_coef_1,Max_Q_coef_2,...,Max_Q_coef_N]
        tchout_list: 当前运行各台冷机的出水温度
        tsup_set :供水温度设定值

        Returns
        cons_tuple ：约束函数元组
        -------
        """
    # Q2:优化的约束条件是哪些，翻译一下
        # 下限
        beta = 0.3
        # 函数tuple
        cons_tuple = ()
        NX = len(chiller_list)

        def cons_chiller_i_highbound(chiller_class_i, tchout_i, tchin_i, Vch_i, tcdin_i, Max_Q_coef_list):
            """
            获取冷机的约束函数
            Parameters
            ----------
            chiller_class_i :冷机的类
            tchout_i :运行冷机的冷冻水出水温度
            tchin_i :运行冷机的冷冻水进水温度
            Vch_i ：运行冷机的冷冻水流量
            tcdin_i ：运行冷机的冷却水进水温度
            Max_Q_coef_list :各台运行冷机的最大制冷量模型系数[Max_Q_coef_1,Max_Q_coef_2,...,Max_Q_coef_N]

            Returns
            delta： 冷机制冷量与最大制冷量的差值
            -------

            """
            Q_i = chiller_class_i.cal_Q([tchin_i, tchout_i, Vch_i])
            Max_Q_coef_i = Max_Q_coef_list[i]
            Max_Q_i = chiller_class_i.Max_Q_fun([tcdin_i, tchin_i], Max_Q_coef_i)
            delta = (Q_i - Max_Q_i)

            return delta

        def cons_chiller_i_lowbound(chiller_class_i, tchout_i, tchin_i, Vch_i):
            """
            获取冷机的约束函数
            Parameters
            ----------
            chiller_class_i :冷机的类
            tchout_i :冷机的冷冻水出水温度
            tchin_i :冷机的冷冻水进水温度
            Vch_i ：冷机的冷冻水流量

            Returns
            delta： 冷机喘振时制冷量与冷机制冷量的差值
            -------

            """
            Q_i = chiller_class_i.cal_Q([tchin_i, tchout_i, Vch_i])
            delta = (beta * chiller_class_i.Cap - Q_i)

            return delta

        def cons_total(CL, chiller_list, tchout_list, tchin_list, Vch_list, tcdin_list, Max_Q_coef_list):
            """
            获取冷机的约束函数
            Parameters
            ----------
            chiller_list :各台运行冷机的类[CH_1,CH_2,...,CH_N]
            tchout_list :各台运行冷机冷冻水出水温度[tchout_1,tchout_2,...,tchout_N]
            tchin_list :各台运行冷机的冷冻水进水温度[tchin_1,tchin_2,...,tchin_N]
            Vch_list :各台运行冷机的冷冻水流量[Vch_1.Vch_2,...,Vch_N]
            tcdin_list :各台运行冷机的冷却水进水温度[tcdin_1.tcdin_2,...,tcdin_N]
            Max_Q_coef_list :各台运行冷机的最大制冷量模型系数[Max_Q_coef_1,Max_Q_coef_2,...,Max_Q_coef_N]
            Returns
            delta： 末端总冷负荷与冷机子系统制冷量的差值
            -------

            """

            Q = 0
            for i in range(len(chiller_list)):
                chiller_i = chiller_list[i]
                tchout_i = tchout_list[i]
                tchin_i = tchin_list[i]
                Vch_i = Vch_list[i]

                Q_i = chiller_i.cal_Q([tchin_i, tchout_i, Vch_i])

                Q += Q_i

            delta = (CL - Q)

            return delta

        def cons_high(X):
            t = cons_chiller_i_highbound(chiller_class_i, X[i], tchin_i, Vch_i, tcdin_i, Max_Q_coef_list)
            return t

        def cons_low(X):
            t = cons_chiller_i_lowbound(chiller_class_i, X[i], tchin_i, Vch_i)
            return t

        def cons_bound(X):
            t = cons_total(CL_total, chiller_list, X, tchin_list, Vch_list, tcdin_list, Max_Q_coef_list)
            return t

        def cons_tsup_set(X):
            t = self.cons_tsup_set(chiller_list, Vch_list, tsup_set, tchout_list, X) # self.xxx 引用的是class级别下的同名函数
            return t

        for i in range(NX):
            chiller_class_i = chiller_list[i]
            tchin_i = tchin_list[i]
            Vch_i = Vch_list[i]
            tcdin_i = tcdin_list[i]

            func_High = cons_high
            cons_tuple += (func_High,)
            func_Low = cons_low
            cons_tuple += (func_Low,)

        func_total = cons_bound
        func_tsup_set = cons_tsup_set
        cons_tuple += (func_total, func_tsup_set,)

        return cons_tuple

    def _get_bounds(self, tchout_set_UL_bound_list, tchout_set_LL_bound_list, ratio_load_list, tchout_set_current_list):
        """
        获取寻优变量上下限，已通过
        Parameters
        ----------
        tchout_set_UL_bound_list : web端设定的各台冷机出水温度设定值上限[tchout_set_UL_bound_1,tchout_set_UL_bound_2,...,tchout_set_UL_bound_N]
        tchout_set_LL_bound_list : web端设定的各台冷机出水温度设定值下限[tchout_set_LL_bound_1,tchout_set_LL_bound_2,...,tchout_set_LL_bound_N]
        ratio_load_list : 各台冷机的负载率[ratio_load_1, ratio_load_2,...,ratio_load_N]
        tchout_set_current_list :各台冷机冷冻水出水温度设定值[tchout_set_1,tchout_set_2,...,tchout_set_N]

        Returns
        ub :寻优上限
        lb ：寻优下限
        -------

        """

        ub = tchout_set_UL_bound_list
        lb = tchout_set_LL_bound_list

        return ub, lb

    def get_subSYS_load_opt_result(self, Input_para_dict):
        """
        获得优化后的各冷机冷冻水出水温度设定值
        Parameters
        ----------
        Input_para_dict :输入参数-dict格式
        {
        on_off_status_current_list :当前冷机运行状态[on_off_status_1,on_off_status_2,...,on_off_status_N]
        CL_total :末端负荷
        tchin_list :各台冷机的冷冻水进水温度[tchin_1,tchin_2,...,tchin_N]
        tchout_list :各台冷机冷冻水出水温度[tchout_1,tchout_2,...,tchout_N]
        Vch_list :各台冷机的冷冻水流量[Vch_1.Vch_2,...,Vch_N]
        tcdin_list :各台冷机的冷却水进水温度[tcdin_1,tcdin_2,...,tcdin_N]
        Power_coef_list :各台冷机的功率模型系数[Power_coef_1,Power_coef_2,...,Power_coef_N]
        Max_Q_coef_list :各台冷机的最大制冷量模型系数[Max_Q_coef_1,Max_Q_coef_2,...,Max_Q_coef_N]
        chiller_list :各台冷机的类[CH_1,CH_2,...,CH_N]
        tsup_set :供水温度设定值
        tchout_set_current_list :各台冷机冷冻水出水温度设定值[tchout_set_1,tchout_set_2,...,tchout_set_N]
        tchout_set_UL_bound :各台冷机的出水温度设定值寻优上限[tchout_set_UL_bound_1,tchout_set_UL_bound_2,...,tchout_set_UL_bound_N]
        tchout_set_LL_bound :各台冷机的出水温度设定值寻优上限[tchout_set_LL_bound_1,tchout_set_LL_bound_2,...,tchout_set_LL_bound_N]
        opt_step :各台冷机出水温度设定值寻优步长[opt_step_1,opt_step_2,...,opt_step_N]
        ratio_load_list :各台冷机的负载率[ratio_load_1,ratio_load_2,...,ratio_load_N]
        }

        Returns
        tchout_set_opt_list :各台冷机的冷冻水出水温度优化设定值
        -------

        """

        tsup_set = Input_para_dict["tsup_set"]
        CL_total = Input_para_dict["CL_total"]
        total_tchout_set_current_list = Input_para_dict['tchout_set_current_list']
        total_on_off_status_current_list = Input_para_dict['on_off_status_current_list']
        # 获得运行冷机的数据
        Input_para_df = pd.DataFrame(Input_para_dict)
        run_chiller_Input_para_dict = Input_para_df[Input_para_df["on_off_status_current_list"] == 1].to_dict('list')
        on_off_status_current_list = run_chiller_Input_para_dict["on_off_status_current_list"]
        tchin_list = run_chiller_Input_para_dict["tchin_list"]
        tchout_list = run_chiller_Input_para_dict["tchout_list"]
        Vch_list = run_chiller_Input_para_dict["Vch_list"]
        tcdin_list = run_chiller_Input_para_dict["tcdin_list"]
        Power_coef_list = run_chiller_Input_para_dict["Power_coef_list"]
        Max_Q_coef_list = run_chiller_Input_para_dict["Max_Q_coef_list"]
        chiller_list = run_chiller_Input_para_dict["chiller_list"]
        tchout_set_current_list = run_chiller_Input_para_dict["tchout_set_current_list"]
        tchout_set_UL_bound = run_chiller_Input_para_dict["tchout_set_UL_bound"]
        tchout_set_LL_bound = run_chiller_Input_para_dict["tchout_set_LL_bound"]
        opt_step = run_chiller_Input_para_dict["opt_step"]
        ratio_load_list = run_chiller_Input_para_dict["ratio_load_list"]

        N = len(chiller_list)

        # 冷机运行台数
        on_num = sum(on_off_status_current_list)
        if on_num == 1:
            tchout_set_opt_list = tchout_set_current_list
            ESR_opt = 0
        else:

            power_sum = self.subSYS_Power_func(tchin_list, tchout_set_current_list, Vch_list, tcdin_list,
                                               Power_coef_list, chiller_list)
            ub, lb = self._get_bounds(tchout_set_UL_bound, tchout_set_LL_bound, ratio_load_list,
                                      tchout_set_current_list)
            _cons_func = self._get_cons_fuc(chiller_list, CL_total, tchin_list, Vch_list, tcdin_list, Max_Q_coef_list,
                                            tchout_set_current_list, tsup_set)
            # 获取优化目标函数的函数句柄
            _opt_func = self._get_opt_func(tchin_list, Vch_list, tcdin_list, Power_coef_list, chiller_list)
            # 优化步长
            opt_step = opt_step
            # PSO算法优化
            self.Res = PSO(func=_opt_func, dim=N, pop=100, max_iter=1000, lb=lb, ub=ub, w=0.8, c1=0.1, c2=0.1,
                           constraint_ueq=_cons_func)
            self.Res.record_mode = True
            #避免pso寻到的最优值，没有当前设定值节能，令pso寻优的初始最佳值等于当前设定值
            self.Res.gbest_x = tchout_set_current_list
            self.Res.gbest_y = power_sum
            self.Res.run()

            tchout_set_opt_list = list(self.Res.gbest_x)
            power_sum_opt = float(self.Res.gbest_y)
            # 出水温度设定值优化后对应的节能率
            ESR_opt = (power_sum - power_sum_opt) / power_sum * 100

            # 判断最大节能率是否大于设定节能率下限，如果节能率太小，不调整出水温度设定值
            if (ESR_opt <= self.ESR_LL) or (ESR_opt > self.ESR_UL):
                tchout_set_opt_list = tchout_set_current_list
                ESR_opt = 0

        # 所有冷机的出水温度设定值
        for i in range(len(total_on_off_status_current_list)):
            on_off_status_current_i = total_on_off_status_current_list[i]
            total_tchout_set_current_i = total_tchout_set_current_list[i]
            if on_off_status_current_i == 0:
                tchout_set_opt_list.insert(i, total_tchout_set_current_i)

        tchout_set_opt_dict = {}
        # 所有冷机出水温度设定值code
        tchout_set_chiller_code_list = list(map(lambda x: "tchout_set_chiller_" + x, self.total_Chiller_code_list))
        for i in range(len(tchout_set_chiller_code_list)):
            tchout_set_opt_i_dict = {tchout_set_chiller_code_list[i]: float(tchout_set_opt_list[i])}
            tchout_set_opt_dict.update(tchout_set_opt_i_dict)

        # 转为json格式
        tchout_set_opt_json = json.dumps(tchout_set_opt_dict)

        return tchout_set_opt_json, ESR_opt

def get_subSYS_opt_no_onff_result(Input_para_dict):
    """
    获得冷机启停优化结果和出水温度设定值优化结果
    Parameters
    ----------
    Input_para_dict :输入参数-dict格式
    {
    on_off_status_current_list :当前各台冷机运行状态[on_off_status_current_1,on_off_status_current_2,...,on_off_status_current_N]
    CL_total :末端负荷
    CL_total_lastopt :上一优化时刻的末端冷负荷
    tchin_list :各台冷机的冷冻水进水温度[tchin_1,tchin_2,...,tchin_N]
    tchout_list :各台冷机冷冻水出水温度[tchout_1,tchout_2,...,tchout_N]
    Vch_list :各台冷机的冷冻水流量[Vch_1.Vch_2,...,Vch_N]
    tcdin_list :各台冷机的冷却水进水温度[tcdin_1,tcdin_2,...,tcdin_N]
    Power_coef_list :各台冷机的功率模型系数[Power_coef_1,Power_coef_2,...,Power_coef_N]
    Max_Q_coef_list :各台冷机的最大制冷量模型系数[Max_Q_coef_1,Max_Q_coef_2,...,Max_Q_coef_N]
    chiller_list :各台冷机的类[CH_1,CH_2,...,CH_N]
    tsup_set :供水温度设定值
    tchout_set_current_list :当前各台冷机冷冻水出水温度设定值[tchout_set_1,tchout_set_2,...,tchout_set_N]
    tchout_set_UL_bound :各台冷机的出水温度设定值寻优上限[tchout_set_UL_bound_1,tchout_set_UL_bound_2,...,tchout_set_UL_bound_N]
    tchout_set_LL_bound :各台冷机的出水温度设定值寻优上限[tchout_set_LL_bound_1,tchout_set_LL_bound_2,...,tchout_set_LL_bound_N]
    opt_step :各台冷机出水温度设定值寻优步长[opt_step_1,opt_step_2,...,opt_step_N]
    ratio_load_list :各台冷机的负载率[ratio_load_1,ratio_load_2,...,ratio_load_N]
    tsup:总管供水温度

    Returns
    result_json:{
    on_off_chiller_code :各台冷机的启停状态优化结果(0-关闭，1-开启)
    tchout_set_chiller_code :各台冷机的冷冻水出水温度优化设定值
    ONOFF :启停变化模式（0-启停状态不变；-1-减开1台；1-增开1台；2-换开1台）
    OL :负荷优化模式（0-未进行负荷优化；1-进行负荷优化）
    CL_total_lastopt ：上次进行负荷优化时的末端冷负荷
    }
    -------
    """
    chiller_list = Input_para_dict["chiller_list"]
    CL_total = Input_para_dict["CL_total"]
    CL_total_lastopt = Input_para_dict["CL_total_lastopt"]
    on_off_status_current_list = Input_para_dict["on_off_status_current_list"]
    tchout_set_current_list = Input_para_dict["tchout_set_current_list"]
    tchin_list = Input_para_dict["tchin_list"]
    Vch_list = Input_para_dict["Vch_list"]
    tcdin_list = Input_para_dict["tcdin_list"]
    Power_coef_list = Input_para_dict["Power_coef_list"]
    tsup_set = Input_para_dict["tsup_set"]

    # 启停变化模式
    # Q1： 现在算法是不会优化冷机的启停状况的吗
    ONOFF = 0  # 冷机启停状态不变
    # 冷机台数
    Num = len(chiller_list)
    CL_coef = 0.1  # 总负荷变化率下限，大于该变化率下限，进行负荷分配优化
    # 当前时刻与上一优化时刻的末端冷负荷变化量
    delta_CL_total = abs(CL_total - CL_total_lastopt)
    # 当前运行冷机中，最小的冷机的额定制冷量
    Cap_list = []
    for i in range(Num):
        chiller_i = chiller_list[i]
        Cap_i = chiller_i.Cap
        Cap_list.append(Cap_i)
    temp_Cap_list = list(np.array(Cap_list) * np.array(on_off_status_current_list))
    if sum(temp_Cap_list) == 0: # 当所有冷机都不运行时，temp_Cap_list等于[0,0,0...,0],
        min_chiller_Cap = min(Cap_list)
        message = f"Warning! 没有冷机运行！"
    else:
        min_chiller_Cap = min(filter(lambda x: x > 0, temp_Cap_list))
        message = f"存在冷机运行！"
    print(message)

    delta_CL_total_ratio = delta_CL_total / min_chiller_Cap

    # 冷机子系统定义
    muti_chillers_SYS = Chillers_CL_allocate_system(chiller_list)
    ESR_LL_set = muti_chillers_SYS.ESR_LL
    ESR_UL_set = muti_chillers_SYS.ESR_UL

    if delta_CL_total_ratio > CL_coef:
        tchout_set_opt_json, ESR_opt = muti_chillers_SYS.get_subSYS_load_opt_result(Input_para_dict)
        tchout_set_opt_dict = json.loads(tchout_set_opt_json)
        tchout_set_opt_list = list(tchout_set_opt_dict.values())

        if (ESR_opt <= ESR_LL_set) or (ESR_opt > ESR_UL_set):
            OL = 0
            no_load_opt_reason = "未满足设定的节能率要求"
        else:
            OL = 1
            no_load_opt_reason = "进行了负荷优化"
            CL_total_lastopt = CL_total
    else:
        OL = 0
        ESR_opt = 0
        tchout_set_opt_list = tchout_set_current_list
        no_load_opt_reason = "未满足设定的负荷变化下限"

    # 出水温度设定值保存1位小数
    tchout_set_opt_list = list(map(lambda x: round(x, 1), tchout_set_opt_list))
    on_off_status_opt_list = on_off_status_current_list

    # ============================================节能率ESR计算：与出水温度全部设为7.5℃比较===============================================#

    Power = 0
    Power_opt = 0
    for i in range(len(chiller_list)):
        Chiller_i = chiller_list[i]
        Power_coef_i = Power_coef_list[i]
        tchin_i = tchin_list[i]
        tchout_set_i = tsup_set
        tchout_set_opt_i = tchout_set_opt_list[i]
        Vch_i = Vch_list[i]
        tcdin_i = tcdin_list[i]
        on_off_status_current_i = on_off_status_current_list[i]

        input_para = [tchin_i, tchout_set_i, Vch_i, tcdin_i]
        Power_i = Chiller_i.Power_cond_fun(input_para, Power_coef_i) * on_off_status_current_i
        Power = Power + Power_i

        input_para_opt = [tchin_i, tchout_set_opt_i, Vch_i, tcdin_i]
        Power_opt_i = Chiller_i.Power_cond_fun(input_para_opt, Power_coef_i) * on_off_status_current_i
        Power_opt = Power_opt + Power_opt_i

    ESR = (Power - Power_opt) / Power * 100

    if ESR < 0:
        ESR = 0

    # ============================================================================================================================#

    on_off_chiller_code_list = []
    # 出水温度设定值code
    tchout_set_chiller_code_list = []

    # 获取所有冷机的列表
    total_Chiller_code_list = []
    for chiller in chiller_list:
        total_Chiller_code_list.append(chiller.device_name)

    for code in total_Chiller_code_list:
        on_off_chiller_code_list.append(code + "_status_switch_opt")
        tchout_set_chiller_code_list.append(code + "_temp_set_chw_out_opt")
    on_off_status_opt_dict = {}
    tchout_set_opt_dict = {}
    for i in range(len(tchout_set_chiller_code_list)):
        on_off_status_opt_i_dict = {on_off_chiller_code_list[i]: float(on_off_status_opt_list[i])}
        tchout_set_opt_i_dict = {tchout_set_chiller_code_list[i]: float(tchout_set_opt_list[i])}
        on_off_status_opt_dict.update(on_off_status_opt_i_dict)
        tchout_set_opt_dict.update(tchout_set_opt_i_dict)

    result_dict = {}
    result_dict.update(on_off_status_opt_dict)
    result_dict.update(tchout_set_opt_dict)
    result_dict.update({"ONOFF": ONOFF})
    result_dict.update({"OL": OL})
    result_dict.update({"CL_total_lastopt": CL_total_lastopt})
    result_dict.update({"ESR": ESR})
    result_dict.update({"ESR_opt": ESR_opt})
    result_dict.update({"delta_CL_total_ratio": delta_CL_total_ratio})
    result_dict.update({"CL_coef_set": CL_coef})
    result_dict.update({"ESR_LL_set": ESR_LL_set})
    result_dict.update({"ESR_UL_set": ESR_UL_set})
    result_dict.update({"no_load_opt_reason": no_load_opt_reason})
    result_dict.update({"message": message})

    # 转为json格式

    result_json = json.dumps(result_dict)

    return result_json


def read_device_data(
        conn=conn_000012,
        paras_list=["status_switch"],
        start='2022-11-02',
        end='2023-04-04',
        table='dech_3051_l1'):
    """
    spread and flatten the table
    :param conn: connection conneected to pg database
    :param paras_list: parameter list Default chiller paras
    :param start: start time
    :param end: deprecated
    :param table: table name Default dech_174_l1
    :return: the flattened table
    """
    paras = ', '.join(paras_list)
    sql = f'''select data_time,device_name,{paras} from {table} where data_time >= '{start}' '''
    data = pd.read_sql(sql, conn)
    # 将设备名和数据类型合并为一列
    df_pivot = pd.pivot_table(data, index='data_time', columns='device_name', values=paras_list, aggfunc='first')
    df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]
    df_pivot.reset_index(inplace=True)

    return df_pivot

def get_power_coef(ch_coef_jason):
    """
    读取冷机模型系数-json格式
    :param ch_coef_jason:
    :return:
    power_coef_list: 各台冷机的功率模型系数
    max_q_coef_list: 各台冷机的最大制冷量模型系数
    """
    """
    从交大数据库读取模型系数，转为冷机模型函数中，所需的list格式
    :return: coef_jason: 冷机模型系数
    """
    coef_dict = json.loads(ch_coef_jason["coef"])
    power_coef_dict_list = coef_dict["power_coef"]
    power_coef_list = []

    max_q_coef_dict_list = coef_dict["cooling_max_coef"]
    max_q_coef_list = []
    for i in range(len(power_coef_dict_list)):


        power_coef_i_dict = power_coef_dict_list[i]
        power_coef_i = power_coef_i_dict["data"]
        power_coef_i_list = []
        for j in range(len(power_coef_i)):
            power_coef_i_j_dict = power_coef_i[j]
            power_coef_i_j = power_coef_i_j_dict["data"]
            power_coef_i_list.append(power_coef_i_j)
        power_coef_list.append(power_coef_i_list)

        max_q_coef_i_dict = max_q_coef_dict_list[i]
        max_q_coef_i = max_q_coef_i_dict["data"]
        max_q_coef_list.append(max_q_coef_i)

    return power_coef_list, max_q_coef_list


#======================main_loop===============================================================================================#

# 一次侧总管name
cvpp_name = "decvpp01"

# 室外气象devicename
th = "deth01"

# 冷机name
total_ch = ["dech01", "dech02", "dech03"]

# 实例化各台冷机
chiller01 = Chiller("dech01", "dech01", 1100, 658.2, 5, 184.1)
chiller02 = Chiller("dech02", "dech02", 1100, 658.2, 5, 184.1)
chiller03 = Chiller("dech03", "dech03", 1100, 695.4, 5, 184.1)

# 实例化多冷机系统

chiller_list = [chiller01, chiller02, chiller03]

# 导入冷机模型
ch_coef_json = json.load(open("data_ch/coef/ch_coef.json", "r"))
Power_coef_list, Max_Q_coef_list = get_power_coef(ch_coef_json)

# 优化设定参数
tchout_set_UL_bound_list = [8.5, 7, 8.5]
tchout_set_LL_bound_list = [6, 6, 6]
opt_step = [0.1, 0.1, 0.1]
tsup_set = 7.4
delta_tsup_set = 0.2

status = "actual_run"
# status = "simulate"
time_intveal = pd.Timedelta("20 minutes")

#======================status=simulate=========================================================================================#

if status == "actual_run":
    last_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    last_time = pd.Timestamp(last_time) - pd.Timedelta("30 minutes")  # ensure the first optimization can be conducted
    data_num = 100 * 366 * 60 * 24
elif status == "simulate":
    data = pd.read_excel("simulate_data.xlsx")
    data.fillna(method="ffill",inplace=True)
    data.reset_index(inplace=True)
    data_num = len(data)
    last_time = pd.Timestamp(data.iloc[0:1]['data_time'][0])
    last_time = pd.Timestamp(last_time) - pd.Timedelta("30 minutes")
else:
    Warning("default use simulate mode-------!!!---------------")

for i in range(data_num):
    print(f"----------------第{i}个数据点----开始----------------")
    if i == 0:
        CL_total_lastopt = 0 # 第一次优化时，令上一优化时刻的负荷的0kW，即保证第一次可以被优化
    else:
        # 读取 txt 文件的最后一行数据
        with open("data_ch/optimization_result.txt", "r") as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
        # 解析最后一行数据为 JSON 格式
        last_data = json.loads(last_line)
        # 获取 "CL_total_lastopt" 值
        CL_total_lastopt = last_data["CL_total_lastopt"]
    if status == "simulate":
        data_i = data.iloc[i:i + 1]
    elif status == "actual_run":
        # 为了取时间
        sql_sample_ch = "select data_time from dech_3051_l1 ORDER BY data_time DESC LIMIT 1;"
        data_sample_ch = pd.read_sql(sql_sample_ch, con=conn_000012)
        sql_sample_cvpp = "select data_time from decwp_3051_l1 ORDER BY data_time DESC LIMIT 1;"
        data_sample_cvpp = pd.read_sql(sql_sample_cvpp, con=conn_000012)

        data_sample = min(data_sample_ch["data_time"][0], data_sample_cvpp["data_time"][0])

        data_time_now = data_sample - pd.Timedelta("10 minutes")  # 这个数据采集的频率有问题，为了保险起见，用的是10分钟以内的前的数据
        # 后面改用取最小时间戳的方式解决
        data_time_now = data_time_now.strftime("%Y-%m-%d %H:%M:00")

        # 读取冷机数据
        ch_para_list = ["status_switch", "temp_cow_in", "temp_cow_out", "temp_chw_in", "temp_set_chw_out",
                        "temp_chw_out", "p_evaporator", "p_condenser", "ratio_load",
                        'power_active', 'flow_instantaneous_2', 'flow_instantaneous']
        data_ch = read_device_data(start=data_time_now,
                                   paras_list=ch_para_list,
                                   table="dech_3051_l1")
        # 读取一次侧数据
        cvpp_para_list = ["temp", "temp_2", "flow_instantaneous"]
        data_cvpp = read_device_data(start=data_time_now,
                                     paras_list=cvpp_para_list,
                                     table="decwp_3051_l1")

        data_cvpp["CL_total"] = (data_cvpp[f"{cvpp_name}_temp_2"] - data_cvpp[f"{cvpp_name}_temp"]) * \
                                data_cvpp[f"{cvpp_name}_flow_instantaneous"] * 4.18 * 1000 / 3600 #改成三 和

        # 读取室外气象参数
        th_para_list = ["temp_outdoor", "temp_wb_outdoor"]
        data_th = read_device_data(start=data_time_now,
                                     paras_list=th_para_list,
                                     table="deth_3051_l1")

        data_i_temp = pd.merge(data_ch, data_cvpp, how='outer', on="data_time")
        data_i = pd.merge(data_i_temp, data_th, how='outer', on="data_time")
        data_i_save = data_i.copy()  # 为了debug使用
        data_i.fillna(method='ffill', inplace=True)

        if len(data_i) >= 2:
            data_i = data_i[-2:-1]

    now_time_str = data_i["data_time"].iloc[0]
    now_time = pd.Timestamp(now_time_str)
    if now_time - last_time < time_intveal:
        print(f"-------------------当前时刻为：{now_time_str}----------------------------------------------------")
        print("----------------{:5.0f} 分钟之内不用进行优化---读线程休眠----------------------".format(
            time_intveal.seconds / 60))
        if status == 'actual_run':
            # N分钟后进行再读取
            time.sleep(time_intveal.seconds / 2)
    else:
        print(f"-------------------当前时刻为：{now_time_str}----------------------------------------------------")
        print(
            "----------------距离上一次优化已经超过 {:5.0f} 分钟，进行优化-----------".format(time_intveal.seconds / 60))

        last_time = now_time

        CL_total = data_i["CL_total"].iloc[0]
        tsup = data_i[f"{cvpp_name}_temp"].iloc[0]

        status_switch_list = []
        temp_chw_in_list = []
        temp_chw_out_list = []
        flow_chw_list = []
        temp_cow_in_list = []
        temp_set_chw_out_list = []
        ratio_load_list = []
        power_active_list = []
        temp_cow_out_list = []
        for j, ch_name in enumerate(total_ch):
            status_switch_list.append(int(data_i[f"{ch_name}_status_switch"].iloc[0]))
            temp_chw_in_list.append(data_i[f"{ch_name}_temp_chw_in"].iloc[0])
            temp_chw_out_list.append(data_i[f"{ch_name}_temp_chw_out"].iloc[0])
            flow_chw_list.append(data_i[f"{ch_name}_flow_instantaneous"].iloc[0])
            temp_cow_in_list.append(data_i[f"{ch_name}_temp_cow_in"].iloc[0])
            temp_cow_out_list.append(data_i[f"{ch_name}_temp_cow_out"].iloc[0])
            ratio_load_list.append(data_i[f"{ch_name}_ratio_load"].iloc[0])
            power_active_list.append(data_i[f"{ch_name}_power_active"].iloc[0])
            temp_set_chw_out_list.append(data_i[f"{ch_name}_temp_set_chw_out"].iloc[0])

        # 解决当前当前总管水温与各台冷机换算水温不一致--后续删除
        temp_set_chw_out_list_temp = [value for index, value in enumerate(temp_set_chw_out_list) if status_switch_list[index] != 0]
        tsup_set_temp = np.mean(temp_set_chw_out_list_temp)
        if tsup_set_temp > tsup_set:
            temp_set_chw_out_list = [tsup_set] * len(temp_set_chw_out_list)

        # tsup_set = math.ceil(sum(temp_chw_out_list_temp) / len(temp_chw_out_list_temp) * 10) / 10
        # # 当前采集的出水温度设定值不正确，根据出水温度采集值确定
        # temp_out_set = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
        # temp_set_chw_out_list = [min(temp_out_set, key=lambda x: abs(x - temp_out_set_element)) for temp_out_set_element in temp_chw_out_list]

        #==============actual-run调试时需删除======================#
        if i > 0:
            temp_set_chw_out_list = [last_data[f"{ch_name}_temp_set_chw_out_opt"] for ch_name in total_ch]

            #====判断当前总管供水温度是否过高或过低=====#
            #计算过去10分钟总管供水温度平均值
            tsup_ave = data_i["decvpp01_temp"].mean(axis=0)
            #计算总管供水温度平均值与当前总管供水温度设定值的偏差
            delta_tsup = tsup_ave - tsup_set

            if abs(delta_tsup) > delta_tsup_set:
                tsup_set = tsup_set - delta_tsup

        else:
            # 第一次优化时，对比基准是原始策略
            temp_set_chw_out_list = [tsup_set] * len(chiller_list)

        Input_para_dict = {}
        Input_para_dict['CL_total'] = CL_total
        Input_para_dict['tchin_list'] = temp_chw_in_list
        Input_para_dict['tchout_list'] = temp_chw_out_list
        Input_para_dict['Vch_list'] = flow_chw_list
        Input_para_dict['tcdin_list'] = temp_cow_in_list
        Input_para_dict['tsup'] = tsup
        Input_para_dict['tchout_set_current_list'] = temp_set_chw_out_list
        Input_para_dict['ratio_load_list'] = ratio_load_list
        Input_para_dict['CL_total_lastopt'] = CL_total_lastopt
        Input_para_dict['Power_coef_list'] = Power_coef_list
        Input_para_dict['Max_Q_coef_list'] = Max_Q_coef_list
        Input_para_dict['chiller_list'] = chiller_list
        Input_para_dict['tchout_set_UL_bound'] = tchout_set_UL_bound_list
        Input_para_dict['tchout_set_LL_bound'] = tchout_set_LL_bound_list
        Input_para_dict['opt_step'] = opt_step
        Input_para_dict['tsup_set'] = tsup_set
        Input_para_dict["on_off_status_current_list"] = status_switch_list

        optimize_result_json = get_subSYS_opt_no_onff_result(Input_para_dict)

        # 将JSON字符串写入文件
        with open('data_ch/optimization_result.txt', "a+") as file:
            file.write(optimize_result_json + "\n")

        # ===============================================记录系统数据====================================================#
        optimize_result_dict = json.loads(optimize_result_json)

        power_active_predict_list = []
        power_active_opt_list = []
        temp_set_chw_out_opt_list = []
        status_switch_opt_list = []
        # 计算每台冷机运行功率的仿真值
        for j, ch_name in enumerate(total_ch):
            chiller_j = chiller_list[j]
            power_active_j = power_active_list[j]
            Power_coef_j = Power_coef_list[j]
            temp_chw_in_j = temp_chw_in_list[j]
            temp_set_chw_out_j = temp_chw_out_list[j]
            temp_set_chw_out_opt_j = optimize_result_dict[f"{ch_name}_temp_set_chw_out_opt"]
            flow_chw_j = flow_chw_list[j]
            temp_cow_in_j = temp_cow_in_list[j]
            status_switch_j = status_switch_list[j]
            status_switch_opt_j = optimize_result_dict[f"{ch_name}_status_switch_opt"]

            input_para = [temp_chw_in_j, temp_set_chw_out_j, flow_chw_j, temp_cow_in_j]
            power_active_predict_j = chiller_j.Power_cond_fun(input_para, Power_coef_j) * status_switch_j
            data_i[f"{ch_name}_power_active_predict"] = power_active_predict_j
            data_i[f"{ch_name}_power_active_predict_error"] = (power_active_j - power_active_predict_j) / power_active_j
            power_active_predict_list.append(power_active_predict_j)

            input_para_opt = [temp_chw_in_j, temp_set_chw_out_opt_j, flow_chw_j, temp_cow_in_j]
            power_active_opt_j = chiller_j.Power_cond_fun(input_para_opt, Power_coef_j) * status_switch_opt_j
            data_i[f"{ch_name}_power_active_opt"] = power_active_opt_j
            power_active_opt_list.append(power_active_opt_j)

            temp_set_chw_out_opt_list.append(temp_set_chw_out_opt_j)
            status_switch_opt_list.append(status_switch_opt_j)

            data_i[f"{ch_name}_status_switch_opt"] = optimize_result_dict[f"{ch_name}_status_switch_opt"]
            data_i[f"{ch_name}_temp_set_chw_out_opt"] = optimize_result_dict[f"{ch_name}_temp_set_chw_out_opt"]

        #系统总能耗
        data_i["sys_power_active"] = sum(power_active_list)
        data_i["sys_power_active_predict"] = sum(power_active_predict_list)
        data_i["sys_power_active_opt"] = sum(power_active_opt_list)

        record_data_dict = data_i.iloc[0].to_dict()
        record_data_dict['CL_total_lastopt'] = optimize_result_dict['CL_total_lastopt']
        record_data_dict['load_is_optimize'] = optimize_result_dict['OL']
        record_data_dict['start_change_num'] = optimize_result_dict['ONOFF']
        record_data_dict['ESR'] = optimize_result_dict['ESR']
        record_data_dict['ESR_opt'] = optimize_result_dict['ESR_opt']
        record_data_dict['delta_CL_total_ratio'] = optimize_result_dict['delta_CL_total_ratio']
        record_data_dict['CL_coef_set'] = optimize_result_dict['CL_coef_set']
        record_data_dict['ESR_LL_set'] = optimize_result_dict['ESR_LL_set']
        record_data_dict['ESR_UL_set'] = optimize_result_dict['ESR_UL_set']
        record_data_dict['no_load_opt_reason'] = optimize_result_dict['no_load_opt_reason']
        record_data_dict["data_time"] = record_data_dict["data_time"].strftime("%Y-%m-%d %H:%M:00")
        record_data_dict["local_time"] = time.strftime("%Y-%m-%d %H:%M:00", time.localtime(time.time()))

        json_data = json.dumps(record_data_dict, ensure_ascii=False)  # 保证中文不被编码成ASII
        time_stamp = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        file_name = f"log_{time_stamp}.txt"
        with open(file_name, "a+") as file:  # a+ 而非 w+
            file.write(json_data + "\n")

        #============================打印参数============================================#

        status_switch_json = json.dumps(dict(zip(total_ch, status_switch_list)))
        temp_set_chw_out_json = json.dumps(dict(zip(total_ch, temp_set_chw_out_list)))
        status_switch_opt_json = json.dumps(dict(zip(total_ch, status_switch_opt_list)))
        temp_set_chw_out_opt_json = json.dumps(dict(zip(total_ch, temp_set_chw_out_opt_list)))

        print("------------------------------------工况及优化结果参数--------------------------------")
        print("当前数据： 冷负荷为{:5.2f} kW，    湿球温度为{} ℃ ".format(data_i["CL_total"].iloc[0], data_i[f"{th}_temp_wb_outdoor"].iloc[0]))
        print("当前数据： 冷冻水供水温度{:5.2f} ℃".format(data_i[f"{cvpp_name}_temp"].iloc[0]))
        print("当前数据： 实际的冷机系统总功率为{:5.2f} kW".format(data_i["sys_power_active"].iloc[0]))
        print("当前数据： 预测的冷机系统总功率为{:5.2f} kW".format(data_i["sys_power_active_predict"].iloc[0]))
        print("优化结果： 优化后的冷机系统总功率为{:5.2f} kW".format(data_i["sys_power_active_opt"].iloc[0]))
        print("各台冷机当前数据： 启停状态{}  出水温度设定值{} ℃ ".format(status_switch_json, temp_set_chw_out_json))
        print("优化结果： 启停状态{}  出水温度设定值{} ℃ ".format(status_switch_opt_json, temp_set_chw_out_opt_json))
        print("优化结果： 节能率{:5.2f} %".format(record_data_dict['ESR']))

        rem_ch_code_list = ["ch01", "ch02", "ch03"]
        for j, ch_name in enumerate(total_ch):
            rem_ch_code = rem_ch_code_list[j]
            pre_status_switch = optimize_result_dict[f"{ch_name}_status_switch_opt"]
            pre_temp_set_chw_out = optimize_result_dict[f"{ch_name}_temp_set_chw_out_opt"] * 10
            PLC_controler.upload_chiller_out_temp(sys.argv[1:], rem_ch_code, pre_temp_set_chw_out)  # 传入的温度设定值要*10
            # if pre_status_switch != 0:
            #     PLC_controler.upload_chiller_out_temp(sys.argv[1:], pre_temp_set_chw_out)  # 传入的温度设定值要*10
