import os
# os.chdir(r'D:/algorithm-library-tansuo/team_work')
# os.chdir(r'C:/Users/User/PycharmProjects/algorithm-library-tansuo/qiuxin/冷机负荷分配算法模型(7.5edit)')
# os.chdir(r'D:/工作文档/chiller_load_optimization_提交版本')
import datetime
import logging
import sys
from scipy.optimize import minimize
import geatpy as ea
from sko.PSO import PSO
from sklearn.metrics import mean_absolute_percentage_error
from chiller_class_new import Chiller
from chiller_group_class import Chiller_group
from tools import conn_000012
from ALi_Class import Local_2_AliCloud
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm_notebook
from tqdm import tqdm

# os.chdir(r'/')

def get_system_sql_and_num_dict(system_id, now_minus_10, now_minus_5, dech_num=3, decwp_num=1): #DECWP[待改]
    system_sql_dict = {}
    system_sql_dict['dech'] = '''
        select device_name,
        avg(temp_chw_in) as temp_chw_in,
        avg(temp_chw_out) as temp_chw_out,
        avg(temp_set_chw_out) as temp_set_chw_out,
        avg(temp_cow_in) as temp_cow_in,
        avg(temp_cow_out) as temp_cow_out,
        avg(power_active) as power_active,
        avg(flow_instantaneous) as flow_instantaneous,
        avg(flow_instantaneous_2) as flow_instantaneous_2,
        avg(p_condenser) as p_condenser,
        avg(p_evaporator) as p_evaporator,
        avg(ratio_load) as ratio_load,
        avg(status_switch) as status_switch
        from dech_{}_l1
        where device_name LIKE 'dech%' and
        data_time between '{}'and '{}'
        group by device_name
        '''.format(system_id, now_minus_10, now_minus_5)
    system_sql_dict['decwp'] = '''
        select device_name,
        avg(temp) as temp,
        avg(temp_2) as temp_2,
        avg(temp-temp_2) as delta_temp,
        avg(flow_instantaneous) as flow_instantaneous,
        avg((temp_2-temp)*flow_instantaneous*4.18*1000/3600) as CL_total 
        from decwp_{}_l1
        where data_time between '{}' and '{}'
        group by device_name
        '''.format(system_id, now_minus_10, now_minus_5)
    # system_sql_dict['deth'] = '''
    # select device_name,
    # avg(temp_outdoor) as temp_outdoor,
    # avg(temp_wb_outdoor) as temp_wb_outdoor
    # from deth_{}_l1
    # where data_time between '{}' and '{}'
    # group by device_name
    #     '''.format(system_id, now_minus_10, now_minus_5)
    system_num_dict = {}
    system_num_dict['dech'] = dech_num
    system_num_dict['decwp'] = decwp_num
    # system_num_dict['deth'] = deth_num
    return system_sql_dict, system_num_dict

class Optimizer_moudle:
    def __init__(self, system_id, chiller_list: list,
                 simulate_time: pd.Timestamp):
        # 获取system_id
        self.system_id = system_id
        self.simulate_time_str = simulate_time.strftime('%Y-%m-%d %H:%M:00')
        # 获取当前时间-5分钟
        self.simulate_time_minus_5 = (
            simulate_time -
            pd.Timedelta(
                minutes=5)).strftime('%Y-%m-%d %H:%M:00')
        # 获取当前时间-10分钟
        self.simulate_time_minus_10 = (
            simulate_time -
            pd.Timedelta(
                minutes=10)).strftime('%Y-%m-%d %H:%M:00')
        # 获取拼接sql 获取设备数量
        system_sql_dict, system_num_dict = get_system_sql_and_num_dict(
            system_id, self.simulate_time_minus_10, self.simulate_time_minus_5)
        self.system_sql_dict = system_sql_dict
        self.system_num_dict = system_num_dict
        # 获取冷机list
        self.chiller_list = chiller_list
        # 获取上传Ali的接口
        self.Up_load_moudle = Local_2_AliCloud
        # 实例化冷机组
        self.Chiller_group_moudle = Chiller_group(chiller_list=chiller_list,)

    def get_system_dict(self):
        # 对各个系统取出每个系统的frame
        system_dict = {}
        # 会计算是否取到了所有数据（并进行判断）
        judge = 'Pass'
        for i in ['dech', 'decwp']:
            system_dict[i] = {}
            system_dict[i]['num'] = self.system_num_dict[i]
            system_dict[i]['sql'] = self.system_sql_dict[i]
            system_dict[i]['data'] = pd.read_sql(
                con=conn_000012, sql=self.system_sql_dict[i]).dropna(axis=1, how='any')
            system_dict[i]['data']['data_time'] = self.simulate_time_minus_5
            system_dict[i]['data'] = system_dict[i]['data'].dropna(
                axis=1, how='any')
            # 判断行数是否和台数相同
            current_num = system_dict[i]['data'].shape[0]
            system_num = system_dict[i]['num']
            # judge = 'Fall' if current_num != system_num else 'Pass'
            print('{}数据{}/{}'.format(i, current_num, system_num))
        # 经过一整个循环 只有'dech','decwp'这些系统都Pass，judge值才会是'Pass'，否则是'Fall'
        logging.info('数据校验结果为:{}'.format(judge))
        return system_dict, judge

    def get_current_data(self):
        system_dict, judge = self.get_system_dict()
        if judge == 'Fall':
            logging.info('数据校验结果不通过，不进行优化')
            return pd.DataFrame(), judge
        else:
            system_frame = pd.DataFrame()
            for i in ['dech', 'decwp']:
                tmp_frame = pd.pivot_table(
                    system_dict[i]['data'],
                    index='data_time',
                    columns='device_name',
                    aggfunc='first')
                # 修改列的命名
                tmp_frame.columns = [
                    f'{col[1]}_{col[0]}' for col in tmp_frame.columns]
                # 水平方向拼接
                system_frame = pd.concat([system_frame, tmp_frame], axis=1)
            #### 正则表达式筛选变量
            ## 取冷机相关的组合变量
            # 获取冷却流量和
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'flow_instantaneous_2'))
            system_frame['dech_flow_instantaneous_2_sum'] = tmp.sum(axis=1)
            # 获取冷冻流量和
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'flow_instantaneous'))
            system_frame['dech_flow_instantaneous_sum'] = tmp.sum(axis=1)
            # 获取电量和
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'power_active')).values
            system_frame['dech_power_active_sum'] = tmp.sum(axis=1)
            # 计算冷机冷却水的进出温差（冷机侧）
            cow_out = system_frame.filter( # 筛选出复合的列
                regex='^{}.*{}$'.format('(dech|dedch)', 'temp_cow_out')).values[0] # [0]是因为目前是嵌套的二维数组 [[30 32 30]]中取出[30 32 30]
            cow_in = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'temp_cow_in')).values[0]
            flow_2 = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'flow_instantaneous_2')).values[0]
                # ？因为dech1,2,3是并联的，我们要的是总管上的一个温度，所以只能这样去把三个融合成一个。再加/减总支管误差。（偏差修正）
            system_frame['dech_temp_cow_in_all'] = sum(
                cow_in * flow_2) / sum(flow_2)
            system_frame['dech_temp_cow_out_all'] = sum(
                cow_out * flow_2) / sum(flow_2)
            system_frame['dech_temp_cow_delta'] = system_frame['dech_temp_cow_out_all'] - \
                system_frame['dech_temp_cow_in_all']

            chiller_X_dict_power = {}
            chiller_on_off_list = []
            # 计算每台冷机的冷量，并加入X字典，对应的冷机会有对应的字段
            for i in self.Chiller_group_moudle.device_name_list:
                flow = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'flow_instantaneous')).values
                temp_chw_in = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'temp_chw_in')).values
                temp_chw_out = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'temp_chw_out')).values
                power_active = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'power_active')).values
                open = 1 if (
                    flow > 100) and (
                    temp_chw_in -
                    temp_chw_out) > 0 and (
                    power_active > 10) else 0
                system_frame['{}_CL'.format(
                    i)] = flow * (temp_chw_in - temp_chw_out) * 4.1868 * 1000 / 3600
                system_frame['{}_status_switch'.format(i)] = open # [新添]
                system_frame['{}_open'.format(i)] = open
                chiller_on_off_list.append(open)
                # tchin: 冷机冷冻水进水温 tchout: 冷机冷冻水出水温度 Vch: 冷机冷冻水流量 tcdin: 冷机冷却水进水温度
                tmp = system_frame[[
                    # '{}_temp_chw_in'.format(i), '{}_temp_chw_out'.format(i), '{}_flow_instantaneous'.format(i), '{}_temp_cow_in'.format(i)]].values
                    # '{}_CL'.format(i), '{}_p_condenser'.format(i), '{}_p_evaporator'.format(i)]].values
                    '{}_CL'.format(i), '{}_p_condenser'.format(i)]].values
                chiller_X_dict_power[i] = tmp

            # 计算了冷机电量和
            system_frame['dech_power_active_sum_predict'] = \
                self.Chiller_group_moudle.group_power_predict(
                X_dict=chiller_X_dict_power, on_off_list=chiller_on_off_list)

            # 用于修正
            # system_frame['temp_fix_cooling_tower_out'] = system_frame["dech_temp_cow_in_all"] - \
            #     system_frame["decowp01_temp_2"]
            # system_frame['temp_fix_delta_temp'] = system_frame["dech_temp_cow_delta"] - \
            #     system_frame["decowp01_delta_temp"] # decowp01的temp-temp02
            # system_frame['delta_temp_predict'] = system_frame['dech_Qcond_sum'] * 3.6 /\
            #     (4.1868 *
            #      system_frame['decdwp_flow_instantaneous_all_predict'])

            # self.get_log(system_frame)
            return system_frame, judge

    def get_log_system_frame(self, system_frame: pd.DataFrame):
        '''
        这一段主要用来获取日志，获取当前变量和仿真变量的误差，一般来说，误差过大存在两种可能性
        1.系统没有稳定
        2.模型不准缺
        解决方法：
        1.加大优化间隔时间
        2.定期训练模型
        :param system_frame:
        :return:
        '''
        logging.info('数据开始时间：{}'.format(self.simulate_time_minus_10))
        logging.info('数据结束时间：{}'.format(self.simulate_time_minus_5))
        mape_list = []
        true = float(system_frame['dech_power_active_sum'])
        predict = float(system_frame['dech_power_active_sum_predict'])
        mape = mean_absolute_percentage_error(y_true=[true], y_pred=[predict])
        mape_list.append(mape)
        logging.info('当前工况下：冷机实际电耗：{:.2f} kw    冷机模拟电耗：{:.2f}kw    误差为：{:.2f}'.format(
            true,
            predict,
            mape))

    def get_problem(self, system_frame):
        # 获取问题，返回函数，获取目标函数，最小化耗电
        # 这一部分主要是从systemframe中获取当前的一些变量

        # 当前开启数量
        # current_decdwp_open = int(system_frame['decdwp_open'])
        # current_dech_open = int(system_frame['dech_open'])

        @ea.Problem.single
        def evalVars(Vars):  # 定义目标函数（含约束）
            cpw = 4.1868
            tchout_list = Vars # [待改]
            #decdwp_num, decdwp_freq, dect_num, dect_freq = Vars[0], Vars[1], Vars[2], Vars[3]
            # 计算冷机电量
            chiller_X_dict_power = {}
            # chiller_pcond_cons_list = []
            chiller_on_off_list = list(system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'open')).values[0])
            # print(chiller_on_off_list)
            # 计算每台冷机的冷量，并加入X字典
            for i in range(len(self.Chiller_group_moudle.device_name_list)): # [待改]
                device_name = self.Chiller_group_moudle.device_name_list[i]
                # tchin: 冷机冷冻水进水温 tchout: 冷机冷冻水出水温度 Vch: 冷机冷冻水流量 tcdin: 冷机冷却水进水温度
                tmp = np.array([[
                    # float(system_frame['{}_temp_chw_in'.format(device_name)]),
                    # float(tchout_list[i]), # [待改]
                    float(float(system_frame['{}_flow_instantaneous'.format(device_name)]) * (float(system_frame['{}_temp_chw_in'.format(device_name)])-float(tchout_list[i])) * 4.1868 * 1000 / 3600),
                    # float(system_frame['{}_flow_instantaneous'.format(device_name)]),
                    float(system_frame['{}_p_condenser'.format(device_name)])
                    # float(system_frame['{}_p_evaporator'.format(device_name)])
                    ]])
                chiller_X_dict_power[device_name] = tmp
            dech_power = self.Chiller_group_moudle.group_power_predict(chiller_X_dict_power, chiller_on_off_list)

            # 增加了多开、少开冷却塔、冷却泵时的惩罚
            # judge_1 = 10 if decdwp_num != current_decdwp_open else 0
            # judge_2 = 5 if dect_num != current_dect_open else 0
            # power_sum = float(dech_power) + float(decdwp_power) + float(dect_power) + judge_1 +judge_2
            power_sum = float(dech_power)

            f = power_sum  # 计算目标函数值
            print('\n优化后耗电{}'.format(f))
            print('当前耗电{}'.format(float(system_frame['dech_power_active_sum'].values[0])))
            print('当前模拟耗电{}'.format(float(system_frame['dech_power_active_sum_predict'].values[0])))

            CV_list = []
            Q_total = 0
            temp_times_flow_total = 0
            temp_times_flow_acl_total = 0
            for i in range(len(self.Chiller_group_moudle.device_name_list)):  # [待改]
                status = chiller_on_off_list[i]
                if status == 0:
                    continue
                device_name = self.Chiller_group_moudle.device_name_list[i]
                Q_i = float(self.chiller_list[i].Q_calculate(
                    np.array([[
                        float(system_frame['{}_temp_chw_in'.format(device_name)]),
                        float(tchout_list[i]),
                        float(system_frame['{}_flow_instantaneous'.format(device_name)])
                    ]])))
                Q_total += Q_i
                Max_Q_i = float(self.chiller_list[i].Max_Q_predict(
                    np.array([[
                        float(system_frame['{}_temp_chw_in'.format(device_name)]),
                        float(system_frame['{}_temp_cow_in'.format(device_name)])
                    ]])))
                beta = 0.3 # 交大原名原值 便于搜索 待改来源[Nora]
                Min_Q_i = float(self.chiller_list[i].Cap) * beta
                CV_list.append(Q_i - Max_Q_i) # 待确认括号里面是否要加引号 [Nora]
                CV_list.append(Min_Q_i - Q_i)
                # print(CV_list)
                # 补充的 不是交大的
                # CV_list.append(abs(float(tchout_list[i])-float(system_frame['{}_temp_chw_out'.format(device_name)]))-1)
                # CV_list.append(abs(float(tchout_list[i])-float(system_frame['{}_temp_chw_out'.format(device_name)]))-0.5)
                CV_list.append(abs(float(tchout_list[i])-float(system_frame['{}_temp_chw_out'.format(device_name)]))-3)
                temp_times_flow = float(tchout_list[i]) * float(system_frame['{}_flow_instantaneous'.format(device_name)])
                temp_times_flow_acl = float(system_frame['{}_temp_chw_out'.format(device_name)]) * float(system_frame['{}_flow_instantaneous'.format(device_name)])
                temp_times_flow_total += temp_times_flow
                temp_times_flow_acl_total += temp_times_flow_acl

            flow_total = float(system_frame['dech_flow_instantaneous_sum'])
            # tsup_set = 7.4 # 交大原名原值 便于搜索 根据项目修改 [Nora]
            tsup_set = float(system_frame['{}_temp'.format('decvpp01')])
            estimated_temp = temp_times_flow_total / flow_total
            estimated_temp_acl = temp_times_flow_acl_total / flow_total
            #
            corrected_delta = tsup_set - estimated_temp_acl
            corrected_estimated_temp = estimated_temp + corrected_delta
            # print('预计'+str(estimated_temp)) # 修正前8.5左右
            # print('偏差修正后的预计'+str(corrected_estimated_temp))
            # print('实际'+str(tsup_set)) # 修正前7.4左右
            print('设定值')
            print(tchout_list)
            print('优化后出水温度设定值')
            print(corrected_estimated_temp)
            print('当前出水温度设定值')
            print(tsup_set)
            CV_list.append(corrected_estimated_temp - tsup_set)
            # CL_total = (float(system_frame['{}_temp_2'.format('decvpp01')]) - float(system_frame['{}_temp'.format('decvpp01')])) *  \
            #                     float(system_frame['{}_flow_instantaneous'.format('decvpp01')]) * 4.18 * 1000 / 3600 # 不准 用三值之和替换
            CL_total = (float(system_frame['{}_temp_2'.format('decvpp01')]) - float(system_frame['{}_temp'.format('decvpp01')])) * \
                       flow_total * 4.18 * 1000 / 3600  #三值之和
            Q_total = (float(system_frame['{}_temp_2'.format('decvpp01')]) - corrected_estimated_temp) * \
                       flow_total * 4.18 * 1000 / 3600
            CV_list.append(CL_total-Q_total)
            CV_array = np.array(CV_list)
            print(CV_list)
            # print(tchout_list)
            return f, CV_array

        return evalVars

    def solve_problem(self, evalVars, prior_array):
        # 求解器
        problem = ea.Problem(name='Start',
                             M=1,  # 目标维数
                             maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                             # Dim=4,  # 决策变量维数
                             Dim=3,  # 决策变量维数
                             # varTypes=[1, 0, 1, 0],  # 决策变量的类型列表，0：实数；1：整数
                             varTypes=[0, 0, 0],  # 决策变量的类型列表，0：实数；1：整数
                             # 这里后续也要设置成配置的
                             # lb=[2, 30, 3, 20],  # 决策变量下界
                             # ub=[4, 50, 3, 50],  # 决策变量上界 # 暂时水泵上限设置为3
                             lb=[6, 6, 6],
                             ub=[8.5, 7, 8.5],
                             evalVars=evalVars)
        # 选择算法，设置种群数量
        algorithm = ea.soea_SEGA_templet(problem,
                                         ea.Population(Encoding='RI', NIND=300),
                                         MAXGEN=10,  # 最大进化代数。
                                         logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                         trappedValue=1,  # 单目标优化陷入停滞的判断阈值。
                                         maxTrappedCount=1)  # 进化停滞计数器最大上限值。
        res = ea.optimize(algorithm,
                          # 传入当前的工况作为先验知识
                          prophet=prior_array,
                          seed=1, verbose=True,
                          drawing=0, outputMsg=True, drawLog=False,
                          saveFlag=True, dirName='result')
        return res

    def solve(self, up_load=0):
        system_frame, judge = self.get_current_data()
        if judge == 'Fall':
            logging.info('数据缺失，不优化')
            pass
        else:
            self.get_log_system_frame(system_frame)
            current_dech_power_active_sum_predict = float(system_frame['dech_power_active_sum_predict'])

            # 设置问题、求解器
            evalVars = self.get_problem(system_frame)
            # 求解，设置当前工况为先验种群
            current_tchout_list = []
            for i in self.Chiller_group_moudle.device_name_list:
                temp_chw_out = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'temp_chw_out')).values[0][0]
                current_tchout_list.append(temp_chw_out)

            prior_array = np.array(current_tchout_list)
            res = self.solve_problem(evalVars, prior_array)

            log_frame = pd.DataFrame()
            log_frame['data_time'] = [system_frame.index[0]]
            chiller_power_sum = res['ObjV'][0][0]
            tchout_list = res['Vars'][0]
            delta_decvpp_temp = res['CV'][0][-2] # 展示用 [Nora]
            print(res)
            print(delta_decvpp_temp)
            return system_frame, chiller_power_sum, tchout_list, delta_decvpp_temp

# 仿真，过去某个时间的段为输入
def simulate(simulate_time:pd.Timestamp, step = 30,hour = 5):
    log_frame_all = pd.DataFrame()
    system_frame_all = pd.DataFrame()
    for i in range(hour * int(60 / step)):
        time_delta = pd.Timedelta(minutes=step * i)
        run_time = simulate_time + time_delta
        system_id = '3051'
        dech01 = Chiller(system_id, 'dech01', 1100, 1775, 5, 535.6)
        dech02 = Chiller(system_id, 'dech02', 1100, 1775, 5, 535.6)
        dech03 = Chiller(system_id, 'dech03', 1100, 1775, 5, 535.6)
        chiller_list = [dech01, dech02, dech03]
        # now = pd.Timestamp.now()
        aa = Optimizer_moudle('3051', chiller_list, run_time)
        try:
            log_frame, system_frame = aa.solve(up_load=0)
        except:
            logging.info(run_time.strftime('%Y-%m-%d %H:%M:00')+'出现故障')
            log_frame = pd.DataFrame()
            system_frame = pd.DataFrame()
        log_frame_all = log_frame_all.append(log_frame)
        system_frame_all = system_frame_all.append(system_frame)
    return log_frame_all,system_frame_all
# 以当前时间开始仿真
def simulate_solve_now(step=10,hour = 2,up_load = 0):
    log_frame_all = pd.DataFrame()
    system_frame_all = pd.DataFrame()
    for i in range(hour * int(60 / step)):
        time_now = pd.Timestamp.now()
        system_id = '3051'
        dech01 = Chiller(system_id, 'dech01', 1100, 1775, 5, 535.6)
        dech02 = Chiller(system_id, 'dech02', 1100, 1775, 5, 535.6)
        dech03 = Chiller(system_id, 'dech03', 1100, 1775, 5, 535.6)
        chiller_list = [dech01, dech02, dech03]
        aa = Optimizer_moudle('3051', chiller_list, time_now)
        try:
            log_frame, system_frame = aa.solve(up_load)
        except Exception as e:
            logging.info(time_now.strftime('%Y-%m-%d %H:%M:00') + '出现故障')
            log_frame = pd.DataFrame()
            system_frame = pd.DataFrame()
        log_frame_all = log_frame_all.append(log_frame)
        system_frame_all = system_frame_all.append(system_frame)
        time.sleep(60 * step)
            # print()
    return log_frame_all, system_frame_all

def Record_df(simulate_time:pd.Timestamp, hour,step=30,show_progress=True): #不带默认值的参数要写在有默认值的参数之前
    results_df = pd.DataFrame(
        columns = [# 'chiller_power_sum'
                   'data_time', 'dech01_tchout_pre', 'dech02_tchout_pre', 'dech03_tchout_pre',
                   'dech01_tchout_acl', 'dech02_tchout_acl', 'dech03_tchout_acl', 'dech01_tchin', 'dech02_tchin',
                   'dech03_tchin', 'dech01_flow', 'dech02_flow', 'dech03_flow', 'dech01_CL_pre', 'dech02_CL_pre',
                   'dech03_CL_pre', 'dech01_CL_acl', 'dech02_CL_acl', 'dech03_CL_acl', 'total_CL_pre', 'total_CL_acl',
                   'power_active_sum_acl', 'power_active_sum_simu', 'power_active_sum_pre',
                   # 'decvpp_tchout_pre', 'decvpp_tchout_acl',
                   'delta_decvpp_temp',
                   'dech01_CL_acl_ratio',
                   'dech02_CL_acl_ratio',
                   'dech03_CL_acl_ratio',
                   'dech01_CL_pre_ratio',
                   'dech02_CL_pre_ratio',
                   'dech03_CL_pre_ratio'
                   ])
    # for i in range(hour * int(60 / step)):
    # with tqdm_notebook(total=hour * int(60 / step), desc='Processing') as pbar:
    for i in tqdm(range(hour * int(60 / step))):
        time_delta = pd.Timedelta(minutes=step * i)
        run_time = simulate_time + time_delta
        system_id = '3051'
        dech01 = Chiller(system_id, 'dech01', 1100, 1775, 5, 535.6)
        dech02 = Chiller(system_id, 'dech02', 1100, 1775, 5, 535.6)
        dech03 = Chiller(system_id, 'dech03', 1100, 1775, 5, 535.6)
        chiller_list = [dech01, dech02, dech03]
        # now = pd.Timestamp.now()
        aa = Optimizer_moudle('3051', chiller_list, run_time)
        try:
            # log_frame, system_frame = aa.solve(up_load=0)
            # print(aa.solve(up_load=0))
            sys_df = aa.solve()[0]
            chiller_power_sum = aa.solve()[1]
            tchout_list = aa.solve()[2]
            delta_decvpp_temp = aa.solve()[3]
            dech01_CL_acl = sys_df['dech01_flow_instantaneous'].values[0] *(sys_df['dech01_temp_chw_in'].values[0]-sys_df['dech01_temp_chw_out'].values[0]) * 4.1868 * 1000 / 3600
            dech02_CL_acl = sys_df['dech02_flow_instantaneous'].values[0] *(sys_df['dech02_temp_chw_in'].values[0]-sys_df['dech02_temp_chw_out'].values[0]) * 4.1868 * 1000 / 3600
            dech03_CL_acl = sys_df['dech03_flow_instantaneous'].values[0] *(sys_df['dech03_temp_chw_in'].values[0]-sys_df['dech03_temp_chw_out'].values[0]) * 4.1868 * 1000 / 3600
            dech01_CL_pre = sys_df['dech01_flow_instantaneous'].values[0] * (
                        sys_df['dech01_temp_chw_in'].values[0] - tchout_list[0]) * 4.1868 * 1000 / 3600
            dech02_CL_pre = sys_df['dech02_flow_instantaneous'].values[0] * (
                        sys_df['dech02_temp_chw_in'].values[0] - tchout_list[1]) * 4.1868 * 1000 / 3600
            dech03_CL_pre = sys_df['dech03_flow_instantaneous'].values[0] * (
                        sys_df['dech03_temp_chw_in'].values[0] - tchout_list[2]) * 4.1868 * 1000 / 3600
            total_CL_acl = dech01_CL_acl + dech02_CL_acl + dech03_CL_acl
            total_CL_pre = dech01_CL_pre + dech02_CL_pre + dech03_CL_pre
            dech_Cap = 1100 * 3.517
            EER = float(abs(chiller_power_sum-sys_df['dech_power_active_sum'].values[0]) / sys_df['dech_power_active_sum_predict'].values[0])
            results_df = results_df.append({
                'data_time': run_time,
                # 'chiller_power_sum': chiller_power_sum,
                'dech01_tchout_pre': tchout_list[0],
                'dech02_tchout_pre': tchout_list[1],
                'dech03_tchout_pre': tchout_list[2],
                'dech01_tchout_acl': sys_df['dech01_temp_chw_out'].values[0],
                'dech02_tchout_acl': sys_df['dech02_temp_chw_out'].values[0],
                'dech03_tchout_acl': sys_df['dech03_temp_chw_out'].values[0],
                # 'decvpp_tchout_pre': sys_df['decvpp01_temp'].values[0]+delta_decvpp_temp,
                # 'decvpp_tchout_acl': sys_df['decvpp01_temp'].values[0],
                'delta_decvpp_temp': delta_decvpp_temp,
                'dech01_tchin': sys_df['dech01_temp_chw_in'].values[0],
                'dech02_tchin': sys_df['dech02_temp_chw_in'].values[0],
                'dech03_tchin': sys_df['dech03_temp_chw_in'].values[0],
                'dech01_flow': sys_df['dech01_flow_instantaneous'].values[0],
                'dech02_flow': sys_df['dech02_flow_instantaneous'].values[0],
                'dech03_flow': sys_df['dech03_flow_instantaneous'].values[0],
                'dech01_CL_acl': dech01_CL_acl,
                'dech02_CL_acl': dech02_CL_acl,
                'dech03_CL_acl': dech03_CL_acl,
                'dech01_CL_acl_ratio': float(dech01_CL_acl / dech_Cap),
                'dech02_CL_acl_ratio': float(dech02_CL_acl / dech_Cap),
                'dech03_CL_acl_ratio': float(dech03_CL_acl / dech_Cap),
                'dech01_CL_pre': dech01_CL_pre,
                'dech02_CL_pre': dech02_CL_pre,
                'dech03_CL_pre': dech03_CL_pre,
                'dech01_CL_pre_ratio': float(dech01_CL_pre/dech_Cap),
                'dech02_CL_pre_ratio': float(dech02_CL_pre/dech_Cap),
                'dech03_CL_pre_ratio': float(dech03_CL_pre/dech_Cap),
                'total_CL_acl': total_CL_acl,
                'total_CL_pre': total_CL_pre,
                'power_active_sum_acl': sys_df['dech_power_active_sum'].values[0],
                'power_active_sum_simu': sys_df['dech_power_active_sum_predict'].values[0],
                'power_active_sum_pre': chiller_power_sum,
                'EER': EER
            }, ignore_index=True)
        except:
            # logging.info(run_time.strftime('%Y-%m-%d %H:%M:00')+'出现故障')
            # log_frame = pd.DataFrame()
            # system_frame = pd.DataFrame()
            print(run_time.strftime('%Y-%m-%d %H:%M:00')+'出现故障')
    return results_df


if __name__ == "__main__":

    # time_now = pd.Timestamp.now()
    # time_now = pd.Timestamp(year=2023, month=6, day=28, hour=0)
    #
    # system_id = '3051'
    #
    # dech01 = Chiller(system_id, 'dech01', 1100, 1775, 5, 535.6)
    # dech02 = Chiller(system_id, 'dech02', 1100, 1775, 5, 535.6)
    # dech03 = Chiller(system_id, 'dech03', 1100, 1775, 5, 535.6)
    #
    # chiller_list = [dech01, dech02, dech03]
    #
    # aa = Optimizer_moudle('3051', chiller_list, time_now)
    #
    # print(aa.solve(up_load=0))

    ######################################################################

    # filename = 'result_data.csv'
    # if filename in os.listdir():
    #     df = pd.read_csv('result_data.csv', parse_dates=['data_time'])
    # else:
    df = Record_df(hour=8, step=60, simulate_time=pd.Timestamp(year=2023, month=6, day=28, hour=0), show_progress=True)
    df.to_csv('result_data.csv', index=False)
    # df = pd.read_csv('result_data.csv', parse_dates=['data_time'])

    df['data_time'] = pd.to_datetime(df['data_time'])
    # df['data_time'] = df['data_time'].dt.strftime('%H:%M')
    df['data_time'] = df['data_time'].dt.strftime('%m-%d %H:%M')

    for i, j in [
                ['dech01_CL_acl', 'dech01_CL_pre'], ['dech02_CL_acl', 'dech02_CL_pre'], ['dech03_CL_acl', 'dech03_CL_pre'],
                ['total_CL_acl', 'total_CL_pre'], ['power_active_sum_acl', 'power_active_sum_simu'],
                ['delta_decvpp_temp', 'delta_decvpp_temp'],
                ['EER', 'EER'],
                ['dech01_tchout_acl', 'dech01_tchout_pre'],
                ['dech02_tchout_acl', 'dech02_tchout_pre'],
                ['dech03_tchout_acl', 'dech03_tchout_pre']
                ]:
        # 绘制图形
        fig, ax1 = plt.subplots()  # 左边的纵坐标

        ax1.plot(df['data_time'], df[i], label=i)
        if j != 'delta_decvpp_temp' and j != 'EER':
            ax1.plot(df['data_time'], df[j], label=j)
        if i == 'power_active_sum_acl':
            ax1.plot(df['data_time'], df['power_active_sum_pre'], label='power_active_sum_pre')
        if i.startswith('dech') and 'CL' in i:
            ax2 = ax1.twinx()  # 右边的纵坐标
            ax2.plot(df['data_time'], df[i + '_ratio'], label=i + '_ratio')
            ax2.plot(df['data_time'], df[i.replace("_acl", "") + "_pre_ratio"], label=i.replace("_acl", "") + "_pre_ratio")
            ax2.legend(loc='upper right')

        ax1.set_xlabel('Time')
        # ax1.set_ylabel('CL')

        ax1.legend(loc='upper left')
        # plt.title('CL vs. Time')

        # 设置 x 轴刻度
        x_ticks = np.arange(len(df['data_time']))
        # 仅显示奇数索引位置的刻度标签
        ax1.set_xticks(x_ticks[::8])
        # ax2.set_xticks(x_ticks[::8])

        # 自动调整日期标签
        fig.autofmt_xdate()

        # 设置背景透明
        # plt.gca().set_facecolor('none')

        # 保存图像为PNG文件（带有透明背景）
        # plt.savefig('transparent_line_plot.png', transparent=True)
        plt.show()

