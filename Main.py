from datetime import datetime, timedelta
from email.mime.text import MIMEText
import hashlib
import random
import re
import smtplib
import subprocess
from waitress import serve # type: ignore
from flask import Flask, jsonify,request # type: ignore
from flask_cors import CORS # type: ignore
from wxpusher import WxPusher # type: ignore
import requests # type: ignore
import pymysql # type: ignore
import time
import os
import json
import shutil
import psutil # type: ignore
from os.path import join
import importlib.util
import string
from rcon import Client # type: ignore
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler # type: ignore
from sparkai.core.messages import ChatMessage # type: ignore
from openai import OpenAI # type: ignore
from zhipuai import ZhipuAI # type: ignore
import qianfan # type: ignore
import base64

app = Flask(__name__)

help_text = """
/help 获取帮助
/审核 [玩家名] 将玩家加入白名单
/时间 获取当前时间
bilibili search [名称] 搜索视频名称
bing search [名称] 使用必应搜索
/status 获取当前服务器状态
签到 签到
商店 查看商店
模型列表 查看模型列表
切换模型 [模型名称] 切换为指定模型"""
status_help = """
/status指令可用条目：
运行内存：查询服务器总内存
CPU：查询服务器CPU占用率
运行时长：查询服务器运行时长"""
rcon_host = '127.0.0.1'
rcon_port = 25575
rcon_password = ''
api_ip = 'http://127.0.0.1:3000'
whitelist = {}
without_ban = []

# AI部分

model_list = ['spark-lite','讯飞星火','讯飞星火-lite','hunyuan','腾讯混元','hunyuan-lite','智谱清言','glm','chatglm','文心一言','文心一言-speed','ERNIE Speed','ERNIE-Speed-128K','ERNIE-Speed','文心一言-lite','ERNIE Lite','ERNIE-Lite-8K','ERNIE-Lite','文心一言-tiny','ERNIE Tiny','ERNIE-Tiny','ERNIE-Tiny-8K','deepseek','DeepSeek','DeepSeek-R1','DeepSeek-8B','深度思考','deepseek-r1','qwen','Qwen','通义千问','Qwen2.5','Tongyi']

'''讯飞星火'''
xing_model = ['spark-lite','讯飞星火-lite','讯飞星火']
#星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'
#星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = ''
SPARKAI_API_SECRET = ''
SPARKAI_API_KEY = ''
#星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'lite'

'''腾讯混元'''
tencent_model = ['腾讯混元','hunyuan','hunyuan-lite']
Tencent_AppKey = ""

'''智谱清言'''
zhipu_key = ''
zhipu_model = ['智谱清言','glm','chatglm']

'''文心一言'''
baidu_model_speed = ['文心一言','文心一言-speed','ERNIE Speed','ERNIE-Speed-128K','ERNIE-Speed']
baidu_model_lite = ['文心一言-lite','ERNIE Lite','ERNIE-Lite-8K','ERNIE-Lite']
baidu_model_tiny = ['文心一言-tiny','ERNIE Tiny','ERNIE-Tiny','ERNIE-Tiny-8K']

os.environ["QIANFAN_ACCESS_KEY"] = ""
os.environ["QIANFAN_SECRET_KEY"] = ""

'''硅基流动'''
silicon_key = ''

deepseek_model = ['deepseek','DeepSeek','DeepSeek-R1','DeepSeek-8B','深度思考','deepseek-r1']
qwen_model = ['qwen','Qwen','通义千问','Qwen2.5','Tongyi']

model_group = {}

def encode_to_base64(input_string):
    """
    将输入字符串编码为Base64格式。

    参数:
    input_string (str): 要编码的字符串。

    返回:
    str: Base64编码后的字符串。
    """
    # 将字符串编码为字节
    byte_data = input_string.encode('utf-8')
    # 进行Base64编码
    base64_encoded = base64.b64encode(byte_data)
    # 将Base64编码的字节转换为字符串
    base64_string = base64_encoded.decode('utf-8')
    
    return base64_string

# 调用AI并获取返回
def NoStreamChat(model, self):
    if model in xing_model:
        spark = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=SPARKAI_APP_ID,  # 直接传递参数
            spark_api_key=SPARKAI_API_KEY,
            spark_api_secret=SPARKAI_API_SECRET,
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
        )
        messages = [ChatMessage(
            role="user",
            content=self
        )]
        handler = ChunkPrintHandler()
        try:
            a = spark.generate([messages], callbacks=[handler])
        except Exception as e:
            print("Error:", e)
            return "Error: " + str(e)
        print("Generated text:", a.generations[0][0].text)  # 添加调试信息
        return str(a.generations[0][0].text)
    elif model in tencent_model:
        client = OpenAI(
            api_key=Tencent_AppKey,  # 混元 APIKey
            base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
        )
        completion = client.chat.completions.create(
            model='hunyuan-lite',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        print("Generated text:", completion.choices[0].message.content)  # 添加调试信息
        return str(completion.choices[0].message.content)
    elif model in zhipu_model:
        client = ZhipuAI(api_key=zhipu_key)
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {
                        "role": "user",
                        "content": self,
                    },
                ],
                extra_body={},
            )
            print("Generated text:", response.choices[0].message.content)  # 添加调试信息
            return str(response.choices[0].message.content)
        except Exception as e:
            print("Error:", e)
            return "Error: " + str(e)
    elif model in baidu_model_lite:
        chat_comp = qianfan.ChatCompletion()
        # 指定特定模型
        resp = chat_comp.do(model="ERNIE-Lite-8K", messages=[{
            "role": "user",
            "content": self
        }])
        print("Generated text:",resp["body"])
        return resp["body"]['result']
    elif model in baidu_model_speed:
        chat_comp = qianfan.ChatCompletion()
        resp = chat_comp.do(model="ERNIE-Speed-128K", messages=[{
            "role": "user",
            "content": self
        }])
        print("Generated text:",resp["body"])
        return resp["body"]['result']
    elif model in baidu_model_tiny:
        chat_comp = qianfan.ChatCompletion()
        resp = chat_comp.do(model="ERNIE-Tiny-8K", messages=[{
            "role": "user",
            "content": self
        }])
        print("Generated text:",resp["body"])
        return resp["body"]['result']
    elif model in deepseek_model:
        client = OpenAI(
            api_key=silicon_key,  
            base_url="https://api.siliconflow.cn/v1",
        )
        completion = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        print("Generated text:", completion.choices[0].message.content)
        return str(completion.choices[0].message.content)
    elif model in deepseek_model:
        client = OpenAI(
            api_key=silicon_key,  
            base_url="https://api.siliconflow.cn/v1",
        )
        completion = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        print("Generated text:", completion.choices[0].message.content)
        return str(completion.choices[0].message.content)
    elif model in qwen_model:
        client = OpenAI(
            api_key=silicon_key,  
            base_url="https://api.siliconflow.cn/v1",
        )
        completion = client.chat.completions.create(
            model='Qwen/Qwen2.5-Coder-7B-Instruct',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        print("Generated text:", completion.choices[0].message.content)
        return str(completion.choices[0].message.content)

StreamInfo = {}
StreamChatNum = 0

def ApplyStreamId(data,model):
    if model not in model_list:
        return -1
    global StreamChatNum
    global StreamInfo
    StreamChatNum += 1
    if data['group_id'] not in StreamInfo:
        StreamInfo[data['group_id']] = {}
        StreamInfo[data['group_id']][data['user_id']] = {}
        StreamInfo[data['group_id']][data['user_id']]['StreamId'] = StreamChatNum
        StreamInfo[data['group_id']][data['user_id']]['StreamChat'] = []
        StreamInfo[data['group_id']][data['user_id']]['Model'] = model
        return StreamChatNum
    else:
        if data['user_id'] not in StreamInfo[data['group_id']]:
            StreamInfo[data['group_id']][data['user_id']] = {}
            StreamInfo[data['group_id']][data['user_id']]['StreamId'] = StreamChatNum
            StreamInfo[data['group_id']][data['user_id']]['StreamChat'] = []
        else:
            return StreamInfo[data['group_id']][data['user_id']]['StreamId']

def IsStreamChatExist(data):
    global StreamInfo
    group_id = data['group_id']
    user_id = data['user_id']
    if group_id in StreamInfo and user_id in StreamInfo[group_id]:
        return True
    else:
        return False

def StreamChat(data):
    if not IsStreamChatExist(data):
        return '对话不存在'
    global StreamInfo
    model = StreamInfo[data['group_id']][data['user_id']]['Model']
    if model in tencent_model:
        content = StreamInfo[data['group_id']][data['user_id']]['StreamChat']
        content.append({'role':'user','content':data['message']})
        client = OpenAI(
            api_key=Tencent_AppKey,  # 混元 APIKey
            base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
        )
        completion = client.chat.completions.create(
            model='hunyuan-lite',
            messages=content,
            extra_body={},
        )
        print("Generated text:", completion.choices[0].message.content)  # 添加调试信息
        content.append({'role':'assistant','content':completion.choices[0].message.content})
        print('本次对话后的content',str(content))
        StreamInfo[data['group_id']][data['user_id']]['StreamChat'] = content
        return str(completion.choices[0].message.content)

# WhyAPI部分
WhyKey = ''
def DouyinHot():
    try:
        response = requests.get(f'https://whyta.cn/api/tx/douyinhot?key={WhyKey}')
        response.raise_for_status()
        data = response.json()
        res = ""
        print(data)
        for i in data['result']['list']:
            res += f"{i['word']}\n"
        return res
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

def NetHot():
    try:
        response = requests.get(f'https://whyta.cn/api/tx/networkhot?key={WhyKey}')
        response.raise_for_status()
        data = response.json()
        res = ""
        print(data)
        for i in data['result']['list']:
            res += f"\n{i['title']}"
        return res
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

# SH-API
def BilibiliSearch(name):
    try:
        response = requests.get(f'https://api.yyy001.com/api/blisearch?msg={name}')
        response.raise_for_status()
        data = response.json()['data']
        res = []
        print(data)
        for i in data:
            res += (f"\n{i['id']}.{i['title']}(BVID:{i['bvid']})")
        return res
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

# 起零API
qiling_key = ''

def SearchWithBing(keyword):
    api = "https://api.istero.com/resource/bing/search"
    data = {"token": "96105152f07ece9955147b9e46b92aea","keyword":keyword}
    response = requests.post(api, data=data).json()
    data = response['data']
    msg = ''
    for i in data:
        msg += f"{i['title']}\n"
    return msg

def BanKeyWord(msg):
    try:
        ret = requests.get(f'https://api.yyy001.com/api/Forbidden?text={msg}').json()
        code = ret.get('code')
        banCount = ret.get('data').get('banCount')
        if code != 200:
            return "关键词检测接口异常，请联系开发者"
        if banCount == 0:
            return msg
        else:
            li = list(ret.get('data').get('banList'))
            for i in li:
                k = i.get('word')
                msg = msg.replace(k, '*' * len(k))
            return msg
    except Exception as e:
        print(e)
        return "关键词检测接口异常，请联系开发者"

def SendGroupMsg(data,msg,double=False):
    if data['group_id'] not in without_ban:
        msg = BanKeyWord(msg)
    try:
        if double:
            requests.get(f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false")
        requests.get(f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false")
    except:
        print("发送失败:" + f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false")

def SendPrivateMsg(data,msg):
    try:
        requests.get(f"{api_ip}/send_private_msg?user_id={data['user_id']}&message={msg}&auto_escape=false")
    except:
        print("发送失败:" + f"{api_ip}/send_private_msg?user_id={data['user_id']}&message={msg}&auto_escape=false")

def PushMsg(group_id,msg):
    try:
        requests.get(f"{api_ip}/send_group_msg?group_id={group_id}&message={msg}&auto_escape=false")
    except:
        print("发送失败:" + f"{api_ip}/send_group_msg?group_id={group_id}&message={msg}&auto_escape=false")

def get_cpu_usage():
    """ 获取当前系统的CPU使用率 """
    cpu_percent = psutil.cpu_percent(interval=1)
    return cpu_percent

def SingIn(data):
    # 签到
    with open('./sing.json','r') as f:
        sing_list = json.load(f)
    print(sing_list)
    user_id = str(data['user_id'])
    group_id = str(data['group_id'])
    datetime_str = datetime.now().strftime('%Y-%m-%d')
    print(datetime_str)
    print(group_id in sing_list)
    if group_id in sing_list and user_id in sing_list[group_id]:
        #获取年月日
        print(f'[上次签到]{sing_list[group_id][user_id]}，判断测试{sing_list[group_id][user_id] == datetime_str}')
        if sing_list[group_id][user_id] == datetime_str:
            return '今天已经签到过了'
        else:
            sing_list[group_id][user_id] = datetime_str
            with open('./sing.json','w') as f:
                json.dump(sing_list,f)
            return f'签到成功，获得10金币\n总金币：{AddCoin(data,10)}'
    else:
        if group_id not in sing_list:
            sing_list[group_id] = {}
        if user_id not in sing_list[group_id]:
            sing_list[group_id][user_id] = datetime_str
        with open('./sing.json','w') as f:
            json.dump(sing_list,f)
        return f'签到成功，获得10金币\n总金币：{AddCoin(data,10)}'

def AddCoin(data,coin):
    user_id = data.get('user_id')
    group_id = data.get('group_id')
    with open('./user.json','r') as f:
        data = json.load(f)
    if str(group_id) not in data:
        data[str(group_id)] = {}
    if str(user_id) not in data[str(group_id)]:
        data[str(group_id)][str(user_id)] = {'coin':0}
    data[str(group_id)][str(user_id)]['coin'] += coin
    with open('./user.json','w') as f:
        json.dump(data,f,indent=4)
    return data[str(group_id)][str(user_id)]['coin']

def TestMarkdown():
    PushMsg(487886163,f'主动消息测试')
    data = {
	"markdown": {
		"custom_template_id": "102646446_1740143006",
		    "params": [{
				    "key": "day",
				    "values": ["1"]
			    },
			    {
				    "key": "days",
				    "values": ["1"]
			    },
                {
                    "key": "imgurl",
                    "values": ["https://static.yearnstudio.cn/static/logo.jpg"]
                }
		    ]
	    }
    }
    PushMsg(487886163,encode_to_base64(f'[CQ=markdown,data={json.dumps(data)}]'))

def GetMemAll():
    mem = psutil.virtual_memory()
    return round(mem.total / (1024.0 ** 3),2)

@app.route("/",methods=["POST","GET"])
def root():
    data = request.json
    if 'group_id' not in data:
        msg = str(data['raw_message'])
        if msg == '/时间':
            SendPrivateMsg(data,f"当前时间是{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return {}
        SendPrivateMsg(data,"命令不正确，目前私聊只支持/时间指令")
    print(data)
    msg = str(data['raw_message'])
    if msg == '/help':
        SendGroupMsg(data,help_text)
        return 'Successfully',200
    if msg == '/时间':
        SendGroupMsg(data,f"当前时间是{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 'Successfully',200
    elif msg.startswith("/审核 "):
        SendGroupMsg(data,f"已添加白名单,祝您游戏愉快")
        msg = msg.replace("/审核 ","")
        with Client(rcon_host, rcon_port, passwd=rcon_password) as c:
            response = c.run('whitelist add ' + msg)
            c.run('whitelist save')
            return 'Successfully',200
    else:
        if msg == '添加内测':
            with open('./whitegroup.json','r') as f:
                whitegourp = json.load(f)
            whitegourp[data['group_id']] = 'all'
            with open('./whitegroup.json','w') as f:
                json.dump(whitegourp,f)
            SendGroupMsg(data,f"已将本群添加白名单")
            return 'Successfully',200
        with open('./whitegroup.json','r') as f:
            whitegourp = json.load(f)
        if f"{data['group_id']}" not in whitegourp:
            print(whitegourp)
            SendGroupMsg(data,"命令不正确，目前支持/mc、/审核和/时间指令")
            return 'Successfully',200
    if msg.startswith("/status "):
        msg = msg.replace("/status ","")
        if msg == "运行内存":
            mem = GetMemAll()
            SendGroupMsg(data,f"总内存为{mem}GB")
            return 'Successfully',200
        elif msg == "CPU":
            cpu = get_cpu_usage()
            SendGroupMsg(data,f"CPU占用为{cpu}%")
            return 'Successfully',200
        elif msg == "运行时长":
            # 获取系统启动时间
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            now = datetime.now()
            uptime = now - boot_time
            # 去掉小数点
            uptime = str(uptime).split('.')[0]
            SendGroupMsg(data,f"系统运行时长为{uptime}")
            return 'Successfully',200
        else:
            SendGroupMsg(data,"请输入正确的参数")
            return 'Successfully',200
    elif msg.startswith("/mc "):
        SendGroupMsg(data,"该指令已禁用")
        return 'Successfully',200
        # 去掉开头的/mc
        msg = msg.replace("/mc ","")
        with Client(rcon_host, rcon_port, passwd=rcon_password) as c:
            response = c.run(msg)
            SendGroupMsg(data,response)
            return 'Successfully',200
    elif msg == '本次会话':
        SendGroupMsg(data,f"\nGroup_ID={data['group_id']}\nReal_Group_ID={data['real_group_id']}\nUser_ID={data['user_id']}\nMessage_ID={data['message_id']}\nRaw_Message={data['raw_message']}\nMessage_Type={data['message_type']}\nSub_Type={data['sub_type']}\nFont={data['font']}\nSelf_Id={data['self_id']}\nPost_Type={data['post_type']}\n--End--")
        return 'Successfully',200
    elif msg == '会话':
        SendGroupMsg(data,f"\nGroup_ID={data['group_id']}\nReal_Group_ID={data['real_group_id']}\n--End--")
        return 'Successfully',200
    elif msg == '/status':
        SendGroupMsg(data,status_help)
        return 'Successfully',200
    elif msg == '查询订单':
        SendGroupMsg(data,f"暂无订单")
        return 'Successfully',200
    elif msg.startswith('echo '):
        SendGroupMsg(data,msg.replace('echo ','',1),True)
        return 'Successfully',200
    elif msg.startswith("切换模型 "):
        msg = msg.replace("切换模型 ","")
        if msg in model_list:
            model_group[data['group_id']] = msg
            SendGroupMsg(data,f"已将本群的模型切换为{msg}")
            return 'Successfully',200
        else:
            SendGroupMsg(data,f"模型不存在")
            return 'Successfully',200
    elif msg.startswith("bilibili "):
        msg = msg.replace("bilibili ","")
        if msg.startswith('search '):
            msg = msg.replace('search ','')
            if msg == '':
                SendGroupMsg(data,f"请输入搜索内容")
                return 'Successfully',200
            SendGroupMsg(data,f"正在搜索...")
            ret = BilibiliSearch(msg)
            SendGroupMsg(data,i)
            return 'Successfully',200
    elif msg == '当前模型':
        if data['group_id'] in model_group:
            SendGroupMsg(data,f"当前群组的模型为{model_group[data['group_id']]}")
            return 'Successfully',200
        else:
            SendGroupMsg(data,f"当前群组的模型为spark-lite")
            model_group[data['group_id']] = 'spark-lite'
            return 'Successfully',200
    elif msg == '抖音热榜':
        SendGroupMsg(data,DouyinHot())
        return {}
    elif msg == '全网热榜':
        SendGroupMsg(data,NetHot())
        return {}
    elif msg == '关闭过滤':
        without_ban.append(data['group_id'])
        SendGroupMsg(data,f"已关闭过滤")
        return {}
    elif msg == '模型列表':
        SendGroupMsg(data,f"模型列表:")
        retu = ''
        for i in model_list:
            retu += f"{i}\n"
        retu += f"共{len(model_list)}个模型可用"
        SendGroupMsg(data,retu)
        return {}
    elif msg.startswith('bing search '):
        SendGroupMsg(data,f"正在搜索...")
        msg = msg.replace('bing search ','')
        res = SearchWithBing(msg)
        SendGroupMsg(data,f"  {res}")
        return {}
    elif msg.startswith('开启流式对话 使用模型 '):
        msg = msg.replace('开启流式对话 使用模型 ','')
        SendGroupMsg(data,f"正在申请流式对话...")
        # 申请流式对话
        id = ApplyStreamId(data,msg)
        if id == -1:
            SendGroupMsg(data,f"未知的模型")
            return {}
        SendGroupMsg(data,f"已开启流式对话，编号为{StreamChatNum}")
        return {}
    elif msg == '签到':
        SendGroupMsg(data,SingIn(data))
        return {}
    elif msg == '商店':
        SendGroupMsg(data,'商店为空')
        return {}
    if data['group_id'] not in model_group:
        model_group[data['group_id']] = 'spark-lite'
    # 判断是否为流式对话
    if data['group_id'] in StreamInfo and data['user_id'] in StreamInfo[data['group_id']]:
        SendGroupMsg(data,'正在思考(当前处于流式对话模型)')
        '''流式对话'''
        ret = StreamChat(data)
        SendGroupMsg(data,ret)
        SendGroupMsg(data,'对话完成')
        return {}
    SendGroupMsg(data,f"正在思考...")
    SendGroupMsg(data,NoStreamChat(model_group[data['group_id']],msg))
    SendGroupMsg(data,'对话完成')
    return {}

if __name__ == "__main__":
    with open('./whitelist.json', 'r') as f:
        whitelist = json.load(f)
    # TestMarkdown()
    if not os.path.exists('./user.json'):
        with open('./user.json', 'w') as f:
            json.dump({}, f)
    if not os.path.exists('./sing.json'):
        with open('./sing.json', 'w') as f:
            json.dump({}, f)
    app.run(host='0.0.0.0', port=8090, debug=True, threaded=True)
