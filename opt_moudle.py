import json
import sys
from tools.API import API_class
sys.path.append('/lib/algorithm/3051_lengquece_opt')
# sys.path.append('/lib/tools/algorithm/QQR_env')
# os.chdir(r'D:/algorithm-library-tansuo/team_work')
from timeout_decorator import timeout
#os.chdir(r'C://Users//User//PycharmProjects//algorithm-library-tansuo//team_work')
# os.chdir('/lib/tools/algorithm/QQR_env/team_work')
import logging
import sys
import geatpy as ea
from sklearn.metrics import mean_absolute_percentage_error
from basic_class.chiller_class import Chiller
from basic_class.chiller_group_class import Chiller_group
from basic_class.coolingtower_class import Cooling_tower
from basic_class.pump_class import Pump
from basic_class.pump_group_class import Pump_group
from tools.tools import conn_000012,up_load_log
from basic_class.ALi_Class import Local_2_AliCloud
import numpy as np
import pandas as pd
import time

# os.chdir(r'/')

'''
#配置信息
1.sql
2.项目设备台数
'''


def get_system_sql_and_num_dict(system_id, now_minus_10, now_minus_5, dech_num=10,
                                decdwp_num=10, dect_num=22, decowp_num=1, deth_num=1):#[Nora]
    system_sql_dict = {}
    system_sql_dict['dech'] = '''
        select device_name,
        avg(temp_chw_in) as temp_chw_in,
        avg(temp_chw_out) as temp_chw_out,
        avg(temp_cow_in) as temp_cow_in,
        avg(temp_cow_out) as temp_cow_out,
        avg(power_active) as power_active,
        avg(flow_instantaneous) as flow_instantaneous,
        avg(flow_instantaneous_2) as flow_instantaneous_2,
        avg(p_condenser) as p_condenser,
        avg(p_evaporator) as p_evaporator
        from dech_{}_l1
        where device_name LIKE 'dech%'   and
        data_time between '{}'and '{}'
        group by device_name
        '''.format(system_id, now_minus_10, now_minus_5)
    system_sql_dict['decdwp'] = '''
        select device_name,
        avg(freq) as freq,
        avg(power_active) as power_active,
        max(pr_2 - pr_3) as delta_pr
        from decdwp_{}_l1
        where data_time between '{}' and '{}'
        group by device_name
        '''.format(system_id, now_minus_10, now_minus_5)
    system_sql_dict['dect'] = '''
        select device_name,
        avg(freq) as freq,
        avg(power_active) as power_active
        from dect_{}_l1
        where data_time between '{}' and '{}'
        group by device_name
        '''.format(system_id, now_minus_10, now_minus_5)
    system_sql_dict['decowp'] = '''
        select device_name,
        avg(temp) as temp,
        avg(temp_2) as temp_2,
        avg(temp-temp_2) as delta_temp,
        avg(flow_instantaneous) as flow_instantaneous
        from decowp_{}_l1
        where data_time between '{}' and '{}'
        group by device_name
        '''.format(system_id, now_minus_10, now_minus_5)
    system_sql_dict['deth'] = '''
    select device_name,
    avg(temp_outdoor) as temp_outdoor,
    avg(temp_wb_outdoor) as temp_wb_outdoor
    from deth_{}_l1
    where data_time between '{}' and '{}'
    group by device_name
        '''.format(system_id, now_minus_10, now_minus_5)
    system_num_dict = {}
    system_num_dict['dech'] = dech_num
    system_num_dict['decdwp'] = decdwp_num
    system_num_dict['dect'] = dect_num
    system_num_dict['decowp'] = decowp_num
    system_num_dict['deth'] = deth_num
    return system_sql_dict, system_num_dict


class Optimizer_moudle:
    def __init__(self, system_id,
                 chiller_list: list,
                 pump_list: list,#[Nora]
                 simulate_time: pd.Timestamp):
         # 获取system_id
        self.system_id = system_id
        self.simulate_time_str = simulate_time.strftime('%Y-%m-%d %H:%M:00')
         # 获取当前时间-5分钟
        self.api_class = API_class
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
        #  # 实例化冷却泵组模型（冷却泵组需要获取当前系统system_id，水泵数量，冷机数量）
        # self.Decdwp_moudle = Decdwp_pump_group(self.system_id, 'decdwp_all',
        #                                        self.system_num_dict['decdwp'], self.system_num_dict['dech'])
        self.pump_list = pump_list#[Nora]
        # 实例化冷却塔类
        self.Cooling_tower_moudle = Cooling_tower(self.system_id, 'dect_all',
                                                  self.system_num_dict['dect'], self.system_num_dict['dech'])
        # 获取冷机list
        self.chiller_list = chiller_list

        # 获取上传Ali的接口
        self.Up_load_moudle = Local_2_AliCloud

        # 实例化冷机组
        self.Chiller_group_moudle = Chiller_group(chiller_list=chiller_list,)

        # 实例化冷机组
        self.Pump_group_moudle = Pump_group(pump_list=pump_list, ) #[Nora]

        # 获取上下限设置字典
        self.set_dict = {
            'decdwp_num_min':2,#[Nora]这里要问
            'decdwp_num_max':10,
            'dect_num_min':2,
            'dect_num_max':22,
            'delta_temp_min':3.5,
            'delta_temp_max':5,
            'tower_temp_out_min':15,
            'tower_temp_out_max':30
        }

    def get_system_dict(self):
        # 对各个系统取出每个系统的frame
        system_dict = {}
        # 会计算是否取到了所有数据（并进行判断）
        for name in ['dech', 'decdwp', 'dect', 'decowp', 'deth']:
            system_dict[name] = {}
            system_dict[name]['num'] = self.system_num_dict[name]
            system_dict[name]['sql'] = self.system_sql_dict[name]
            system_dict[name]['data'] = pd.read_sql(
                con=conn_000012, sql=self.system_sql_dict[name]).dropna(axis=1, how='any')
            system_dict[name]['data']['data_time'] = self.simulate_time_minus_5
            system_dict[name]['data'] = system_dict[name]['data'].dropna(
                axis=1, how='any')
            current_num = system_dict[name]['data'].shape[0]
            system_num = system_dict[name]['num']
            judge = 'Fall' if current_num != system_num else 'Pass'
            logging.info('{}数据{}/{}'.format(name, current_num, system_num))
        logging.info('数据校验结果为:{}'.format(judge))
        return system_dict, judge

    def get_current_data(self):
        system_dict, judge = self.get_system_dict()
        if judge == 'Fall':
            logging.info('数据校验结果不通过，不进行优化')
            return pd.DataFrame(), judge
        else:
            system_frame = pd.DataFrame()
            for name in ['dech', 'decdwp', 'dect', 'decowp', 'deth']:
                tmp_frame = pd.pivot_table(
                    system_dict[name]['data'],
                    index='data_time',
                    columns='device_name',
                    aggfunc='first')
                tmp_frame.columns = [
                    f'{col[1]}_{col[0]}' for col in tmp_frame.columns]
                system_frame = pd.concat([system_frame, tmp_frame], axis=1)
            # 正则表达式筛选变量

            # 取冷机相关的组合变量
            # 获取冷机的冷却流量和
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
            #  计算冷机冷却水的进出温差（冷机侧）
            cow_out = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'temp_cow_out')).values[0]
            cow_in = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'temp_cow_in')).values[0]
            flow_2 = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'flow_instantaneous_2')).values[0]
            system_frame['dech_temp_cow_in_all'] = sum(
                cow_in * flow_2) / sum(flow_2)
            system_frame['dech_temp_cow_out_all'] = sum(
                cow_out * flow_2) / sum(flow_2)
            system_frame['dech_temp_cow_delta'] = system_frame['dech_temp_cow_out_all'] - \
                system_frame['dech_temp_cow_in_all']

            # 取冷却塔相关的组合变量
            tmp = system_frame.filter(regex='^{}.*{}$'.format('dect', 'freq'))
            system_frame['dect_freq_sum'] = tmp.sum(axis=1)
            system_frame['dect_open'] = (tmp > 10).sum(axis=1)
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('dect', 'power_active'))
            system_frame['dect_power_active_sum'] = tmp.sum(axis=1)
            system_frame['dect_freq_per'] = system_frame['dect_freq_sum'] / \
                system_frame['dect_open']
            system_frame['dect_power_active_per'] = system_frame['dect_power_active_sum'] / \
                system_frame['dect_open']
            # 取冷却泵相关的组合变量
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('decdwp', 'freq'))
            system_frame['decdwp_freq_sum'] = tmp.sum(axis=1)
            system_frame['decdwp_open'] = (tmp > 20).sum(axis=1)
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('decdwp', 'power_active'))
            system_frame['decdwp_power_active_sum'] = tmp.sum(axis=1)
            system_frame['decdwp_freq_per'] = system_frame['decdwp_freq_sum'] / \
                system_frame['decdwp_open']
            system_frame['decdwp_power_active_per'] = system_frame['decdwp_power_active_sum'] / \
                system_frame['decdwp_open']
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('decdwp', 'delta_pr'))
            system_frame['decdwp_delta_pr_sum'] = tmp[tmp > 0].sum(axis=1)
            system_frame['decdwp_delta_pr_per'] = system_frame['decdwp_delta_pr_sum'] / \
                system_frame['decdwp_open']
            system_frame['decdwp_delta_pr_per'] = tmp.T.max()
            system_frame['decdwp_flow_instantaneou_per'] = system_frame['dech_flow_instantaneous_2_sum'] / \
                system_frame['decdwp_open']
            # self.Cooling_tower_moudle.num = int(system_frame['dect_open'])
            # self.Decdwp_moudle.num = int(system_frame['decdwp_open'])
            # 计算了冷却塔总电量和
            system_frame['dect_power_active_sum_predict'] = self.Cooling_tower_moudle.\
                power_predict(system_frame[['dect_freq_per']].values, int(system_frame['dect_open']))
            # 计算了冷却水泵总电量和
            # system_frame['decdwp_power_active_sum_predict'] = self.Decdwp_moudle.\
            #     power_predict(system_frame[['decdwp_freq_per']].values,int(system_frame['decdwp_open'])) #[Nora]
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
                    temp_chw_out) > 0  and(
                    power_active > 10) else 0
                system_frame['{}_CL'.format(
                    i)] = flow * (temp_chw_in - temp_chw_out) * 4.1868 * 1000 / 3600
                flow_2 = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'flow_instantaneous_2')).values
                temp_cow_in = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'temp_cow_in')).values
                temp_cow_out = system_frame.filter(
                    regex='^{}.*{}$'.format(i, 'temp_cow_out')).values
                system_frame['{}_Qcond'.format(
                    i)] = flow_2 * (temp_cow_out - temp_cow_in) * 4.1868 * 1000 / 3600
                system_frame['{}_open'.format(i)] = open
                chiller_on_off_list.append(open)
                tmp = system_frame[[
                    '{}_CL'.format(i), '{}_p_evaporator'.format(i), '{}_p_condenser'.format(i)]].values
                chiller_X_dict_power[i] = tmp
            # 获取冷凝压力的预测值
            for i in self.Chiller_group_moudle.chiller_list:
                name = i.device_name
                tmp = system_frame[[
                    '{}_Qcond'.format(name), '{}_flow_instantaneous_2'.format(name), 'decowp01_temp_2']]
                system_frame['{}_p_condenser_predict'.format(
                    name)] = i.pcond_predict(tmp)

            # 计算了冷机电量和
            system_frame['dech_power_active_sum_predict'] = \
                self.Chiller_group_moudle.group_power_predict(
                X_dict=chiller_X_dict_power, on_off_list=chiller_on_off_list)

            # 计算了热功率 冷功率 之和
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'Qcond')).sum(axis=1)
            system_frame['dech_Qcond_sum'] = tmp
            tmp = system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'CL')).sum(axis=1)
            system_frame['dech_CL_sum'] = tmp
            # 计算虚拟环境流量
            system_frame['decdwp_flow_instantaneous_all_predict'] = self.Decdwp_moudle.\
                flow_instantaneous_predict(system_frame[[
                                           'decdwp_freq_per'
                                            # , 'decdwp_delta_pr_sum'
                                            ]].values, num=int(system_frame['decdwp_open']))/float(system_frame['decdwp_delta_pr_per'])
            # 计算虚拟冷却塔出塔水温
            system_frame['tower_out_temp_predict'] = self.Cooling_tower_moudle.\
                out_temp_fan_on_predict(system_frame[[
                    # 'dech_Qcond_sum',
                    'dech_flow_instantaneous_2_sum',
                    'dect_freq_sum',
                    # 'decowp01_temp',
                    # 'dect_power_active_sum',
                    # 'dect_freq_per',
                    # 'dect_power_active_per',
                    'deth01_temp_outdoor',
                    'deth01_temp_wb_outdoor'

                    ]])
            # 用于修正
            system_frame['temp_fix_cooling_tower_out'] = system_frame["dech_temp_cow_in_all"] - \
                system_frame["decowp01_temp_2"]

            system_frame['delta_temp_predict'] = system_frame['dech_Qcond_sum'] * 3.6 /\
                (4.1868 *
                 system_frame['decdwp_flow_instantaneous_all_predict'])

            system_frame['temp_fix_delta_temp'] = system_frame["delta_temp_predict"] - \
                system_frame["decowp01_delta_temp"]
            # print(system_frame)

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
        true = float(system_frame['decdwp_power_active_sum'])
        predict = float(system_frame['decdwp_power_active_sum_predict'])
        mape = mean_absolute_percentage_error(y_true=[true], y_pred=[predict])
        mape_list.append(mape)
        logging.info('当前工况下：冷却泵实际电耗： {:.2f} kw    冷却泵模拟电耗：{:.2f}kw    误差为：{:.2f}'.format(
            true,
            predict,
            mape))
        true = float(system_frame['dect_power_active_sum'])
        predict = float(system_frame['dect_power_active_sum_predict'])
        mape = mean_absolute_percentage_error(y_true=[true], y_pred=[predict])
        mape_list.append(mape)
        logging.info('当前工况下：冷却塔实际电耗： {:.2f} kw    冷却塔模拟电耗：{:.2f}kw    误差为：{:.2f}'.format(
            true,
            predict,
            mape))

        true = float(system_frame['dech_power_active_sum'] +
                     system_frame['decdwp_power_active_sum'] +
                     system_frame['dect_power_active_sum'])
        predict = float(system_frame['dech_power_active_sum_predict'] +
                        system_frame['decdwp_power_active_sum_predict'] +
                        system_frame['dect_power_active_sum_predict'])
        mape = mean_absolute_percentage_error(y_true=[true], y_pred=[predict])
        mape_list.append(mape)
        logging.info('当前工况下：冷却侧实际电耗： {:.2f} kw    冷却侧模拟电耗：{:.2f}kw    误差为：{:.2f}'.format(
            true,
            predict,
            mape))

        true = float(system_frame['decowp01_temp_2'])
        predict = float(system_frame['tower_out_temp_predict'])
        mape = mean_absolute_percentage_error(y_true=[true], y_pred=[predict])
        mape_list.append(mape)
        logging.info('当前工况下：实际出塔水温：{:.2f}    冷却侧模拟出塔水温：{:.2f}    误差为{:.2f}'.format(
            true,
            predict,
            mape))

        # 等压力模型做好了再写进去
        for i in self.Chiller_group_moudle.chiller_list:
            name = i.device_name
            true = float(system_frame['{}_p_condenser'.format(name)])
            predict = float(
                system_frame['{}_p_condenser_predict'.format(name)])
            mape = mean_absolute_percentage_error(
                y_true=[true], y_pred=[predict])
            mape_list.append(mape)
            logging.info('{}当前工况下：实际冷机冷凝压力：{:.2f}    冷却侧模拟冷凝压力：{:.2f}    误差为{:.2f}'.format(
                name,
                true,
                predict,
                mape))

        true = float(system_frame['dech_flow_instantaneous_2_sum'])
        predict = float(system_frame['decdwp_flow_instantaneous_all_predict'])
        mape = mean_absolute_percentage_error(y_true=[true], y_pred=[predict])
        mape_list.append(mape)
        logging.info('当前工况下：实际冷却泵组流量：{:.2f}    冷却侧模拟冷却泵组流量：{:.2f}    误差为{:.2f}'.format(
            true,
            predict,
            mape))

        true = float(system_frame['dech_temp_cow_delta'])
        predict = float(system_frame['delta_temp_predict'])
        mape = mean_absolute_percentage_error(y_true=[true], y_pred=[predict])
        tmp = float(system_frame['temp_fix_delta_temp'])
        mape_list.append(mape)
        logging.info('当前工况下：实际冷却泵温差控制：{:.2f}    冷却侧模拟冷却泵组温差控制：{:.2f}    '
                     '误差为{:.2f}    修正量为{:.2f}'.format(
                         true,
                         predict,
                         mape,
                         tmp))

    def get_problem(self, system_frame):
        # 获取问题，返回函数，获取目标函数，最小化耗电
        # 这一部分主要是从systemframe中获取当前的一些变量
        # 当前热功率
        Qcond = float(system_frame['dech_Qcond_sum'])
        # 当前湿球温度
        wb = float(system_frame['deth01_temp_wb_outdoor'])
        out_temp = float(system_frame['deth01_temp_outdoor'])
        # 当前出塔水温
        current_tower_temp_out = float(system_frame['decowp01_temp_2'])
        delta_pr = float(system_frame['decdwp_delta_pr_sum'])
        # 修正温差
        temp_fix_delta_temp = float(system_frame['temp_fix_delta_temp'])
        dect_freq_sum = float(system_frame['dect_freq_sum'])
        decowp01_temp = float(system_frame['decowp01_temp'])
        current_power_active_sum = float(system_frame['dech_power_active_sum'] +
                                         system_frame['decdwp_power_active_sum'] +
                                         system_frame['dect_power_active_sum'])
        current_power_active_sum_predict = float(system_frame['dech_power_active_sum_predict'] +
                                                 system_frame['decdwp_power_active_sum_predict'] +
                                                 system_frame['dect_power_active_sum_predict'])
        # 当前温差设定
        delta_temp_predict_set = float(system_frame['delta_temp_predict'] - temp_fix_delta_temp)
        # 当前开启数量
        current_decdwp_open = int(system_frame['decdwp_open'])
        current_dect_open = int(system_frame['dect_open'])
        # 当前管阻
        decdwp_delta_pr_per = float(system_frame['decdwp_delta_pr_per'])
        # 流量分配模块
        flow_all = system_frame.filter(
            regex='^{}.*{}$'.format('(dech|dedch)', 'flow_instantaneous_2'))
        flow_rate = (flow_all.values/flow_all.values.sum())[0]
        @ea.Problem.single
        def evalVars(Vars):  # 定义目标函数（含约束）
            cpw = 4.1868
            decdwp_num, decdwp_freq, dect_num, dect_freq = Vars[0], Vars[1], Vars[2], Vars[3]
            # 获取模拟流量
            energy = float(self.Decdwp_moudle.flow_instantaneous_predict(np.array([[decdwp_freq
                                                                               # , delta_pr
                                                                            ]]),
                                                                 int(decdwp_num)))
            flow = pow(pow(flow_all.values.sum(), 2) / decdwp_delta_pr_per * energy, 1 / 3)
            flow_list = flow * flow_rate
            # 计算模拟温差设定
            delta_temp = Qcond * 3.6 / (cpw * flow)
            delta_temp_set = float(delta_temp - temp_fix_delta_temp)
            # 计算出塔水温
            tower_temp_out = float(self.Cooling_tower_moudle.out_temp_fan_on_predict(
                np.array([[
                    # Qcond,
                    flow, dect_freq * dect_num
                    , out_temp
                    # ,decowp01_temp
                    , wb]])
            ))
            # 计算冷机电量
            chiller_X_dict_power = {}
            chiller_pcond_cons_list = []
            chiller_on_off_list = list(system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'open')).values[0])
            # 计算每台冷机的冷量，并加入X字典
            for i in range(len(self.Chiller_group_moudle.device_name_list)):
                # print(flow_list[i])
                name = self.Chiller_group_moudle.device_name_list[i]

                tmp_predcit = float(self.chiller_list[i].pcond_predict(
                    np.array([[
                        float(system_frame['{}_Qcond'.format(name)]),
                        flow_list[i],
                        tower_temp_out
                    ]])))
                chiller_pcond_cons_list.append(tmp_predcit)
                tmp = np.array([[
                    float(system_frame['{}_CL'.format(name)]),
                    float(system_frame['{}_p_evaporator'.format(name)]),
                    tmp_predcit]])
                chiller_X_dict_power[name] = tmp
            # print((chiller_X_dict_power, chiller_on_off_list))
            dech_power = self.Chiller_group_moudle.group_power_predict(chiller_X_dict_power, chiller_on_off_list)
            # print(np.array([decdwp_freq]),decdwp_num)
            decdwp_power = self.Decdwp_moudle.power_predict(np.array([decdwp_freq]),decdwp_num)
            # print(np.array([dect_freq]), dect_num)
            dect_power = self.Cooling_tower_moudle.power_predict(np.array([dect_freq]), dect_num)
            # print(dech_power,decdwp_power,dect_power)
            # 增加了多开、少开冷却塔、冷却泵时的惩罚
            judge_1 = 5 if decdwp_num != current_decdwp_open else 0
            judge_2 = 2.5 if dect_num != current_dect_open else 0
            power_sum = float(dech_power) + float(decdwp_power) + float(dect_power) + judge_1 +judge_2
            # f = (current_power_active_sum_predict - power_sum)/current_power_active_sum_predict
            # 最大冷凝压力约束
            max_chiller_pcond = float(max(np.array(chiller_pcond_cons_list)*np.array(chiller_on_off_list)))
            # 逼近度约束
            apporoach = tower_temp_out - wb
            f = power_sum  # 计算目标函数值
            print('系统耗电{}'.format(f))
            print(decdwp_num, decdwp_freq, dect_num, dect_freq )
            # eer = (current_power_active_sum_predict - power_sum)/current_power_active_sum_predict
            # 约束需要根据项目进行更改

            CV = np.array([3 - delta_temp_set, delta_temp_set - 5.2,# 温差控制约束
                           15 - tower_temp_out, tower_temp_out - 31, # 设置出塔水温约束
                           max_chiller_pcond - 145, # 设置pcond最大值约束
                           max(2-apporoach,apporoach-12), #设置冷却塔逼近度约束
                           f-current_power_active_sum_predict
                           # abs(delta_temp_set - delta_temp_predict_set) - 1.5,
                           # abs(tower_temp_out - current_tower_temp_out) - 3,# 设置出塔水温变化幅度约束
                           # abs(dect_num * dect_freq - dect_freq_sum) - 30
                          ])  # 计算违反约束程度
            print(float(dech_power) , float(decdwp_power) , float(dect_power))
            print(decdwp_num, decdwp_freq, dect_num, dect_freq ,tower_temp_out,delta_temp_set)
            print(max_chiller_pcond)
            return f, CV

        return evalVars

    def get_set_point(self, system_frame):
        # 和上面那个函数基本一致，没有差别，用于返回变量
        Qcond = float(system_frame['dech_Qcond_sum'])
        wb = float(system_frame['deth01_temp_wb_outdoor'])
        out_temp = float(system_frame['deth01_temp_outdoor'])
        delta_pr = float(system_frame['decdwp_delta_pr_sum'])
        decowp01_temp = float(system_frame['decowp01_temp'])
        temp_fix_delta_temp = float(system_frame['temp_fix_delta_temp'])
        current_power_active_sum = float(system_frame['dech_power_active_sum'] +
                                         system_frame['decdwp_power_active_sum'] +
                                         system_frame['dect_power_active_sum'])
        current_power_active_sum_predict = float(system_frame['dech_power_active_sum_predict'] +
                                                 system_frame['decdwp_power_active_sum_predict'] +
                                                 system_frame['dect_power_active_sum_predict'])
        delta_temp_predict_set = float(system_frame['delta_temp_predict'] - temp_fix_delta_temp)
        decdwp_delta_pr_per = float(system_frame['decdwp_delta_pr_per'])
        flow_all = system_frame.filter(
            regex='^{}.*{}$'.format('(dech|dedch)', 'flow_instantaneous_2'))
        flow_rate = (flow_all.values/flow_all.values.sum())[0]
        def evalVars(Vars):  # 定义目标函数（含约束）
            cpw = 4.1868
            decdwp_num, decdwp_freq, dect_num, dect_freq = Vars[0], Vars[1], Vars[2], Vars[3]
            energy = float(self.Decdwp_moudle.flow_instantaneous_predict(np.array([[decdwp_freq
                                                                               # , delta_pr
                                                                            ]]),
                                                                 int(decdwp_num)))
            flow = pow(pow(flow_all.values.sum(), 2) / decdwp_delta_pr_per * energy, 1 / 3)
            flow_list = flow * flow_rate
            delta_temp = Qcond * 3.6 / (cpw * flow)
            delta_temp_set = float(delta_temp - temp_fix_delta_temp)
            tower_temp_out = float(self.Cooling_tower_moudle.out_temp_fan_on_predict(
                np.array([[
                    # Qcond,
                    flow, dect_freq * dect_num
                    , out_temp
                    # ,decowp01_temp
                    , wb]])
            ))
            chiller_X_dict_power = {}
            chiller_pcond_cons_list = []
            chiller_on_off_list = list(system_frame.filter(
                regex='^{}.*{}$'.format('(dech|dedch)', 'open')).values[0])
            # 计算每台冷机的冷量，并加入X字典
            for i in range(len(self.Chiller_group_moudle.device_name_list)):
                name = self.Chiller_group_moudle.device_name_list[i]
                tmp_predcit = float(self.chiller_list[i].pcond_predict(
                    np.array([[
                        float(system_frame['{}_Qcond'.format(name)]),
                        flow_list[i],
                        tower_temp_out
                    ]])))
                chiller_pcond_cons_list.append(tmp_predcit)
                tmp = np.array([[
                    float(system_frame['{}_CL'.format(name)]),
                    float(system_frame['{}_p_evaporator'.format(name)]),
                    tmp_predcit]])
                chiller_X_dict_power[name] = tmp
            # print((chiller_X_dict_power, chiller_on_off_list))
            dech_power = self.Chiller_group_moudle.group_power_predict(chiller_X_dict_power, chiller_on_off_list)
            # print(np.array([decdwp_freq]),decdwp_num)
            decdwp_power = self.Decdwp_moudle.power_predict(np.array([decdwp_freq]),decdwp_num)
            # print(np.array([dect_freq]), dect_num)
            dect_power = self.Cooling_tower_moudle.power_predict(np.array([dect_freq]), dect_num)
            # print(dech_power,decdwp_power,dect_power)
            power_sum = float(dech_power) + float(decdwp_power) + float(dect_power)
            f = (current_power_active_sum_predict - power_sum)/current_power_active_sum_predict
            max_chiller_pcond = float(max(chiller_pcond_cons_list))
            return decdwp_num,dect_num,delta_temp_set,tower_temp_out,power_sum,f,dech_power,decdwp_power,dect_power

        return evalVars

    def solve_problem(self,evalVars,prior_array):
        # 求解器，
        problem = ea.Problem(name='Start',
                             M=1,  # 目标维数
                             maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                             Dim=4,  # 决策变量维数
                             varTypes=[1, 0, 1, 0],  # 决策变量的类型列表，0：实数；1：整数
                             # 这里后续也要设置成配置的
                             lb=[2, 30, 3, 20],  # 决策变量下界
                             ub=[3, 50, 3, 50],  # 决策变量上界 # 暂时水泵上限设置为3
                             evalVars=evalVars)
        # 选择算法，设置种群数量
        algorithm = ea.soea_SEGA_templet(problem,
                                         ea.Population(Encoding='RI', NIND=40),
                                         MAXGEN=200,  # 最大进化代数。
                                         logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                         trappedValue=1,  # 单目标优化陷入停滞的判断阈值。
                                         maxTrappedCount=10,
                                         pm = 0.4)  # 进化停滞计数器最大上限值。
        res = ea.optimize(algorithm,
                          # 传入当前的工况作为先验知识
                          prophet = prior_array,
                          seed=1, verbose=True,
                          drawing=1, outputMsg=True, drawLog=True,
                          saveFlag=True, dirName='result')
        return res

    @timeout(60*20)
    def solve(self, up_load = 0):
        system_frame, judge = self.get_current_data()
        if judge == 'Fall':
            logging.info('数据缺失，不优化')
            pass
        else:
            print('进入优化主模块')
            self.get_log_system_frame(system_frame)
            ##
            dech_Qcond_sum = float(system_frame['dech_Qcond_sum'])
            dech_CL_sum = float(system_frame['dech_CL_sum'])
            decdwp_delta_pr_sum = float(system_frame['decdwp_delta_pr_sum'])
            decdwp_flow_instantaneous_all_predict = float(
                system_frame['decdwp_flow_instantaneous_all_predict'])
            temp_fix_delta_temp = float(system_frame['temp_fix_delta_temp'])
            deth01_temp_wb_outdoor = float(
                system_frame['deth01_temp_wb_outdoor'])
            decdwp_freq_per = round(float(system_frame['decdwp_freq_per']), 1)
            dect_freq_per = round(float(system_frame['dect_freq_per']), 1)
            dect_freq_sum = round(float(system_frame['dect_freq_sum']))
            decdwp_freq_sum = round(float(system_frame['decdwp_freq_sum']))
            decowp01_temp_2 = float(system_frame['decowp01_temp_2'])
            decowp01_temp = float(system_frame['decowp01_temp'])
            decowp01_delta_temp = float(system_frame['decowp01_delta_temp'])
            current_decdwp_open = int(system_frame['decdwp_open'])
            current_dect_open = int(system_frame['dect_open'])
            current_dech_power_active_sum_predict = float(system_frame['dech_power_active_sum_predict'])
            current_dect_power_active_sum_predcit = float(system_frame['dect_power_active_sum_predict'])
            current_decdwp_power_active_sum_predict = float(system_frame['decdwp_power_active_sum_predict'])
            current_power_active_sum = float(system_frame['dech_power_active_sum'] +
                                             system_frame['decdwp_power_active_sum'] +
                                             system_frame['dect_power_active_sum'])
            current_power_active_sum_predict = float(system_frame['dech_power_active_sum_predict'] +
                                             system_frame['decdwp_power_active_sum_predict'] +
                                             system_frame['dect_power_active_sum_predict'])
            # 如果增加了一台，则搜索区间是：
            # 如果没有增加，则搜索区间是：
            logging.info('当前工况下：热功率{:.2f}'.format(dech_Qcond_sum))
            logging.info('当前工况下：制冷量{:.2f}'.format(dech_CL_sum))
            logging.info('当前工况下：总管阻{:.2f}'.format(decdwp_delta_pr_sum))
            logging.info('当前工况下：预测流量{:.2f}'.format(
                decdwp_flow_instantaneous_all_predict))
            logging.info('当前工况下：修正温差{:.2f}'.format(temp_fix_delta_temp))
            logging.info('当前工况下：室外湿球{:.2f}'.format(deth01_temp_wb_outdoor))
            logging.info('当前工况下：当前冷却泵频率{:.2f}'.format(decdwp_freq_per))
            logging.info('当前工况下：当前冷却泵开启数量{:.2f}'.format(current_decdwp_open))
            logging.info('当前工况下：当前冷却塔频率{:.2f}'.format(dect_freq_per))
            logging.info('当前工况下：当前冷却塔开启数量{:.2f}'.format(current_dect_open))
            logging.info('当前工况下：当前出塔水温{:.2f}'.format(decowp01_temp_2))
            logging.info('当前工况下：当前温差设定{:.2f}'.format(decowp01_delta_temp))
            # 设置问题、求解器
            evalVars = self.get_problem(system_frame)
            Set_point = self.get_set_point(system_frame)
            # 求解，设置当前工况为先验种群
            prior_array = np.array([[
                current_decdwp_open,
                decdwp_freq_per,
                current_dect_open,
                dect_freq_per
                                     ]])
            print('开始优化')
            res = self.solve_problem(evalVars,prior_array)
            predcit_params = res['Vars'][0]
            # 返回优化结果参数设定值
            decdwp_num_set, dect_num_set, delta_temp_set, \
            tower_temp_out_set, power_sum, f, dech_power_set, \
            decdwp_power_set, dect_power_set = Set_point(predcit_params)
            # decdwp_num, decdwp_freq, dect_num, dect_freq = Vars[0], Vars[1], Vars[2], Vars[3]
            log_frame = pd.DataFrame()
            log_frame['data_time'] = [system_frame.index[0]]

            #存日志，后续维护可以改方案

            log_frame['decdwp_num_set'] = decdwp_num_set
            log_frame['current_decdwp_open'] = current_decdwp_open

            log_frame['decdwp_freq_set'] = predcit_params[1]
            log_frame['current_decdwp_freq'] = decdwp_freq_per

            log_frame['dect_num_set'] = dect_num_set
            log_frame['current_dect_open'] = current_dect_open

            log_frame['dect_freq_set'] = predcit_params[3]
            log_frame['current_dect_freq'] = dect_freq_per

            log_frame['delta_temp_set'] = delta_temp_set
            log_frame['current_delta_temp'] = decowp01_delta_temp

            log_frame['tower_temp_out_set'] = tower_temp_out_set
            log_frame['current_temp_out'] = decowp01_temp_2

            log_frame['dech_power_set'] = dech_power_set
            log_frame['current_dech_power_active_sum_predict'] = current_dech_power_active_sum_predict

            log_frame['decdwp_power_set'] = decdwp_power_set
            log_frame['current_decdwp_power_active_sum_predict']=current_decdwp_power_active_sum_predict

            log_frame['dect_power_set'] = dect_power_set
            log_frame['current_dect_power_active_sum_predict'] = current_dect_power_active_sum_predcit
            log_frame['节能率'] = f
            # 上传参数
            if up_load == 1:
                print('上传寻优参数')
                try:




                    self.Up_load_moudle.upload_decdwp_num(sys.argv[1:],round(decdwp_num_set))
                    self.Up_load_moudle.upload_temp_delta(sys.argv[1:],round(delta_temp_set*10))
                    com = '3051项目:冷却泵{}台，温差设置{}°C'.format(decdwp_num_set,delta_temp_set)
                    self.api_class.commond_post({"systemId":174,"command":com})
                    
                    # 分批上传参数，先上传水泵设置，调整完后再上传冷却塔设置
                    time.sleep(60*5)
                    self.Up_load_moudle.upload_tower_num(sys.argv[1:],round(dect_num_set))
                    self.Up_load_moudle.upload_tower_out_temp(sys.argv[1:],round(tower_temp_out_set*10))
                    com = '3051项目:冷却塔{}台，出塔水温设置{}°C'.format(dect_num_set,tower_temp_out_set)
                    self.api_class.commond_post({"systemId":174,"command":com})

                except:
                    print('上传报错')
            else:
                print('不上传参数')

            return log_frame,system_frame

    def cron_run(self):
        log = pd.DataFrame(columns=['script_start_time', 'script_success',
                                         'script_name', 'script_owner', 'script_cd', 'script_rule',
                                         'script_target', 'script_log', 'remark'], data=[[np.nan for i in range(9)]])
        log['script_name'] = '成都中试线项目 冷却侧优化算法'
        log['script_owner'] = '测试'
        log['script_cd'] = '/lib/tools/algorithm/QQR_env/team_work/cron/cron_optimizer_with.py'
        log['script_rule'] = '5,35 * * * *'
        log['script_start_time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:00')
        try:
            log_frame,system_frame = self.solve(up_load=1)
            log['script_target'] = json.dumps(log_frame.to_dict(), ensure_ascii=False, separators=(',', ':'))
            log['script_log'] = json.dumps(system_frame.to_dict(), ensure_ascii=False, separators=(',', ':')),
            # 记录script_target json
            # 记录script_log json
            log['script_success'] = 1
            log['remark'] = ''
        except Exception as e:
            log['script_target'] = ''
            log['script_log'] = ''
            log['script_success'] = 0
            log['remark'] = str(e)
            print(e)
        up_load_log(log)




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
def simulate_solve_now(step=12,hour = 2,up_load = 0):
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
        try :
            log_frame, system_frame = aa.solve(up_load)
        except Exception as e:
            logging.info(time_now.strftime('%Y-%m-%d %H:%M:00') + '出现故障')
            print(e)
            log_frame = pd.DataFrame()
            system_frame = pd.DataFrame()
        log_frame_all = log_frame_all.append(log_frame)
        system_frame_all = system_frame_all.append(system_frame)
        time.sleep(60*step)
            # print()
    return log_frame_all, system_frame_all


if __name__ == "__main__":
    # 忽略所有警告
    # warnings.simplefilter("ignore",category=np.VisibleDeprecationWarning)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('my_logger')
    logger.info('This is an INFO level message')
# log_frame_all,system_frame_all = simulate(pd.Timestamp(year = 2023,
#                           month=6,
#                           day=8,
#                           hour=12,
#                           minute=0),hour=12)
    # run_time = pd.Timestamp(year=2023,
    #                     month=5,
    #                     day=30,
    #                     hour=0,
    #                     minute=0)
    # time.sleep(60*30)
    # print(os.getcwd())
    # log_frame_all, system_frame_all = simulate_solve_now(step = 30,hour = 13,up_load=1)
    # log_frame_all, system_frame_all = simulate(pd.Timestamp(year = 2023,month=6,day=16), step = 30, hour = 5)
    # solve_frame_all, system_frame_all, best_strategy_all = simulate_solve(simulate_time=time)
    system_id = '3051'
    dech01 = Chiller(system_id, 'dech01', 1100, 1775, 5, 535.6)
    dech02 = Chiller(system_id, 'dech02', 1100, 1775, 5, 535.6)
    dech03 = Chiller(system_id, 'dech03', 1100, 1775, 5, 535.6)
    chiller_list = [dech01, dech02, dech03]
    now = pd.Timestamp.now()
    aa = Optimizer_moudle('3051', chiller_list, now)
    # cron_job = Cron_basic_class(script_name='成都中试线项目 冷却侧优化算法',script_owner='测试',
    #                             script_cd='/lib/tools/algorithm/QQR_env/team_work/cron/cron_optimizer_with.py',
    #                             script_rule='5,35 * * * *',start_time=now)
    aa.solve()
# 装饰器，规定最大运行时间
#     @timeout(1)
#     def job():
#         return aa.solve()
#     cron_job.run(job())
    #
    #
    #
    #
    #
    # try:
    #     log_frame,system_frame = aa.solve(up_load=0)
    #     flag = 1
    #     error_log = ''
    # except Exception as e:
    #     flag = 0
    #     error_log = e
    # log = pd.DataFrame(columns=['script_start_time', 'script_success',
    #                 'script_name','script_owner','script_cd','script_rule',
    #                 'script_target','script_log','remark'],
    #                  data = [[now.strftime('%Y-%m-%d %H:%M:00'),flag,
    #                           '成都中试线项目 冷却侧优化算法','Victor'
    #                              ,'/lib/tools/algorithm/QQR_env/team_work/cron/cron_optimizer_with.py'
    #                              ,'5,35 * * * *',
    #                           json.dumps(log_frame.to_dict(),ensure_ascii=False,separators=(',', ':')),
    #                           json.dumps(system_frame.to_dict(),ensure_ascii=False,separators=(',', ':')),
    #                           error_log]])
    # up_load_log(log)
# solve_frame_all.to_csv('solve_frame_all.csv',encoding='gbk')
# system_frame_all.to_csv('system_frame_all.csv',encoding='gbk')
# best_strategy_all.to_csv('best_strategy_all.csv',encoding='gbk')



