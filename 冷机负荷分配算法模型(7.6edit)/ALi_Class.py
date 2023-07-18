from typing import List

from alibabacloud_iot20180120.client import Client as Iot20180120Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_iot20180120 import models as iot_20180120_models

class Local_2_AliCloud(object):
    def __init__(self):
        pass

    @staticmethod
    def create_client(
            access_key_id: str,
            access_key_secret: str,
    ) -> Iot20180120Client:
        """
        使用AK&SK初始化账号Client
        @param access_key_id:
        @param access_key_secret:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 您的AccessKey ID,
            access_key_id=access_key_id,
            # 您的AccessKey Secret,
            access_key_secret=access_key_secret
        )
        # 访问的域名
        config.endpoint = 'iot.cn-shanghai.aliyuncs.com'
        return Iot20180120Client(config)

    @staticmethod
    def upload_chiller_out_temp(
            args: List[str], chiller_code, chiller_out_temp
    ) -> None:
        client = Local_2_AliCloud.create_client('LTAIIZzJXn8c3YFz', 'rlKLASKPvJXtcVEcRl39JzbRONDwkK')
        set_device_property_request = iot_20180120_models.SetDevicePropertyRequest(
            product_key='a1P5TqkRuVv',
            device_name='PLCCN',
            items='{{"rem_{}_tt_sp":{}}}'.format(chiller_code, chiller_out_temp)  # 有01 02 03 ，对应修改
            # items = '{{"rem_ch01_tt_sp":{}}}'.format(chiller_out_temp)  # 有01 02 03 ，对应修改

        )
        # 复制代码运行请自行打印 API 的返回值
        client.set_device_property(set_device_property_request)
        print(client.set_device_property(set_device_property_request))

    @staticmethod
    def upload_chiller_switch(
            args: List[str], chiller_switch
    ) -> None:
        client = Local_2_AliCloud.create_client('LTAIIZzJXn8c3YFz', 'rlKLASKPvJXtcVEcRl39JzbRONDwkK')
        set_device_property_request = iot_20180120_models.SetDevicePropertyRequest(
            product_key='a1P5TqkRuVv',  # 01 02 03号冷机
            device_name='PLCCN',
            items='{{"rem_chiller_onoff_st_01":{}}}'.format(chiller_switch)
        )
        # 复制代码运行请自行打印 API 的返回值
        client.set_device_property(set_device_property_request)
        print(client.set_device_property(set_device_property_request))