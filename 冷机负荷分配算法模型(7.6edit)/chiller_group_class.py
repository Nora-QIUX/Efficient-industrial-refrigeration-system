import pandas as pd
import numpy as np

class Chiller_group():
    def __init__(self,chiller_list:list):
        self.chiller_list = chiller_list
        self.chiller_num = len(chiller_list)
        self.device_name_list = [i.device_name for i in self.chiller_list]
    def group_power_predict(self,X_dict,on_off_list):
        '''
        :param X_dict:{'dech':pd.DataFrame(对应的输入).values}
        :param on_off_list: [1,1,1]
        :return: 电量合
        '''
        power_list = []
        for i in range(len(self.chiller_list)):
            X = X_dict[self.chiller_list[i].device_name]
            Y = self.chiller_list[i].power_predict(X)
            Y = Y * on_off_list[i]
            power_list.append(Y)
            # power_list.append(Y)
        return np.nan_to_num( np.array(power_list), nan=0).sum(axis = 0)
if __name__ == "__main__":
    dech_dict = {}


