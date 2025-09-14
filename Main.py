from datetime import datetime, timedelta
from email.mime.text import MIMEText
import hashlib
import random
import re
import smtplib
import subprocess
import threading
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
from rcon import Client # type: ignore
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler # type: ignore
from sparkai.core.messages import ChatMessage # type: ignore
from openai import OpenAI # type: ignore
from zhipuai import ZhipuAI # type: ignore
import qianfan # type: ignore
import base64

from apitoken import * # type: ignore

app = Flask(__name__)

help_text = """
/help 获取帮助
/时间 获取当前时间
bilibili search [名称] 搜索视频名称
/status 获取当前服务器状态
签到 签到
商店 查看商店
模型列表 查看模型列表
 注: 可以携带以下参数:
  通义千问
  详细
切换模型 [模型名称] 切换为指定模型
哪咤票房 查看哪咤之魔童闹海的实时票房
weather [城市名] 获取天气信息"""

status_help = """
/status指令可用条目：
运行内存：查询服务器总内存
CPU：查询服务器CPU占用率
运行时长：查询服务器运行时长"""
rcon_host = '127.0.0.1'
rcon_port = 25575
rcon_password = '@Fkchh000'
api_ip = 'http://127.0.0.1:3000'
whitelist = {}
without_ban = []

# 高德API


# AI部分

model_list = ['讯飞星火','腾讯混元','智谱清言','glm','文心一言','DeepSeek','通义千问','ChatGPT','Grok','Grok4','Kimi','Qwen3']

'''讯飞星火'''
xing_model = ['spark-lite','讯飞星火-lite','讯飞星火']
#星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'
#星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
#星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'lite'

'''腾讯混元'''
tencent_model = ['腾讯混元','hunyuan','hunyuan-lite']

'''智谱清言'''

zhipu_model = ['智谱清言','glm','chatglm']

'''文心一言'''
baidu_model_speed = ['文心一言','文心一言-speed','ERNIE Speed','ERNIE-Speed-128K','ERNIE-Speed']
baidu_model_lite = ['文心一言-lite','ERNIE Lite','ERNIE-Lite-8K','ERNIE-Lite']
baidu_model_tiny = ['文心一言-tiny','ERNIE Tiny','ERNIE-Tiny','ERNIE-Tiny-8K']

'''硅基流动'''

deepseek_model = ['deepseek','DeepSeek','DeepSeek-R1','DeepSeek-8B','深度思考','deepseek-r1']
qwen_model = ['qwen','Qwen','通义千问','Qwen2.5','Tongyi']

'''Breath AI'''
BreathAIURL = 'https://api.breathai.top/v1'
chatgpt_model = ['ChatGPT','ChatGPT-5','gpt','chatgpt']
gpt_model_detail = ['gpt-5','gpt-5-chat','gpt-5-mini','gpt-5-nano','gpt-oss-120b','gpt-oss-120b-high','gpt-oss-120b-low','gpt-oss-20b','gpt-oss-20b-high','gpt-oss-20b-low','chatgpt-oss','o3','o3-mini','o4-mini']
grok_model = ['Grok','Grok3','grok','grok3']
grok_4_model = ['Grok4','grok4']
qwen_3_model = ['qwen2.5-vl-72b-instruct','qwen3-14b','qwen3-32b-ultrafast','qwen3-235b-a22b','qwen3-235b-a22b-instruct-2507','qwen3-235b-a22b-thinking-2507','qwen3-30b-a3b','qwen3-30b-a3b-instruct-2507','qwen3-30b-a3b-thinking-2507','qwen3-32b','qwen3-8b','qwen3-coder-30b-a3b-instruct','qwen3-coder-30b-a3b-instruct','qwen3-coder-480b-a33b-instruct','qwenlong-l1-32b','qwq']
kimi_model = ['Kimi','kimi']
deepseek_all_model = ['deepseek-r1','deepseek-r1-0528-qwen3-8b','deepseek-v2.5','deepseek-v3','deepseek-v3.1']
meta_all_model = ['llama-3.1-8b-instant','llama-3.3-70b-versatile','llama-4-maverick','llama-4-scout']
grok_all_model = ['grok-3','grok-3-mini','grok-3-mini-devx','grok-4']
glm_all_model = ['glm-4.5','glm-4.5-air','glm-4.5v']

breath_all_model = gpt_model_detail + qwen_3_model + deepseek_all_model + meta_all_model + ['kimi-k2'] + glm_all_model + ['breath']

model_group = {}

def GetUid(data):
    with open('./user.json','r',encoding='utf-8') as f:
        user = json.load(f)
    user_id = str(data['real_user_id'])
    uid = f"{user_id}"
    if uid not in user:
        user[uid] = {'lashSign':'2000-1-1','SignDays':'0','Coin':'0','model':'qwen3-32b-ultrafast','streamInfo':None,'qid':'0'}
        with open('./user.json','w',encoding='utf-8') as f:
            json.dump(user,f,ensure_ascii=False,indent=4)
        return uid
    elif user[uid]['qid'] != '0':
        uid = user[uid]['qid']
    return uid

def GetZeroCatUser():
    url = 'https://zerocat-api.houlangs.com/api/info'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['user']
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return None


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
        print("[Debug]Call Xing: " + model)
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
        print("[Debug]Call Tencent: " + model)
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
        print("[Debug]Call Zhipu: " + model)
        client = ZhipuAI(api_key=zhipu_key)
        try:
            response = client.chat.completions.create(
                model="glm-4.5-flash",
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
        print("[Debug]Call Baidu Lite: " + model)
        chat_comp = qianfan.ChatCompletion()
        # 指定特定模型
        resp = chat_comp.do(model="ERNIE-Lite-8K", messages=[{
            "role": "user",
            "content": self
        }])
        print("Generated text:",resp["body"])
        return resp["body"]['result']
    elif model in baidu_model_speed:
        print("[Debug]Call Baidu Speed: " + model)
        chat_comp = qianfan.ChatCompletion()
        resp = chat_comp.do(model="ERNIE-Speed-128K", messages=[{
            "role": "user",
            "content": self
        }])
        print("Generated text:",resp["body"])
        return resp["body"]['result']
    
    elif model in chatgpt_model:
        print("[Debug]Call ChatGPT: " + model)
        client = OpenAI(
            api_key=breath_ai_key,   # type: ignore
            base_url=BreathAIURL,
        )
        completion = client.chat.completions.create(
            model='gpt-5',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        # print("Generated text:", completion.choices[0].message.content)
        print(str(completion.choices[0].message.content))
        return str(completion.choices[0].message.content)
    
    elif model == "grok-3-mini-devx":
        print("[Debug]Call Grok Mini Devx: " + model)

        client = OpenAI(
            api_key=breath_ai_key,   # type: ignore
            base_url=BreathAIURL,
        )
        completion = client.chat.completions.create(
            model='grok-3-mini-devx',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        # print("Generated text:", completion.choices[0].message.content)
        print(str(completion.choices[0].message.content))
        return str(completion.choices[0].message.content)
    
    elif model in breath_all_model:

        print("[Debug]Call Breath AI: " + model)
        client = OpenAI(
            api_key=breath_ai_key,   # type: ignore
            base_url=BreathAIURL,
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        # print("Generated text:", completion.choices[0].message.content)
        print("[Debug]re" + str(completion.choices[0].message.content))
        return str(completion.choices[0].message.content)
    
    elif model in grok_model:
        print("[Debug]Call Grok: " + model)

        client = OpenAI(
            api_key=breath_ai_key,   # type: ignore
            base_url=BreathAIURL,
        )
        completion = client.chat.completions.create(
            model='grok-3',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        # print("Generated text:", completion.choices[0].message.content)
        print(str(completion.choices[0].message.content))
        return str(completion.choices[0].message.content)
    
    elif model in grok_4_model:
        print("[Debug]Call Grok 4: " + model)

        client = OpenAI(
            api_key=breath_ai_key,   # type: ignore
            base_url=BreathAIURL,
        )
        completion = client.chat.completions.create(
            model='grok-4',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        # print("Generated text:", completion.choices[0].message.content)
        print(str(completion))
        return str(completion.choices[0].message.content)
    
    elif model in kimi_model:
        print("[Debug]Call Kimi: " + model)

        client = OpenAI(
            api_key=breath_ai_key,   # type: ignore
            base_url=BreathAIURL,
        )
        completion = client.chat.completions.create(
            model='kimi-k2',
            messages=[
                {
                    "role": "user",
                    "content": self,
                },
            ],
            extra_body={},
        )
        # print("Generated text:", completion.choices[0].message.content)
        print(str(completion))
        return str(completion.choices[0].message.content)
    
    elif model in baidu_model_tiny:
        print("[Debug]Call Baidu Tiny: " + model)
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
    elif model in qwen_model:
        print("[Debug]Call Qwen: " + model)
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

def BindQQ(data,qq):
    uid = GetUid(data)
    with open('./user.json','r',encoding='utf-8') as f:
        user = json.load(f)
    user[qq] = user[uid]
    user[uid] = {'qid':qq}
    with open('./user.json','w',encoding='utf-8') as f:
        json.dump(user,f,ensure_ascii=False,indent=4)

def GetBoxUser():
    ret = requests.get('https://sbox.yearnstudio.cn/number_of_users')
    if ret.status_code == 200:
        return ret.json()['number_of_users']
    else:
        return '获取失败'


def GetModel(data):
    uid = GetUid(data)
    with open('./user.json','r',encoding='utf-8') as f:
        user = json.load(f)
    return user[uid]['model']

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
        res = ""
        print(data)
        for i in data:
            res += (f"\n{i['id']}.{i['title']}(BVID:{i['bvid']})")
        return res
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

def DomainInfo(domain):
    try:
        response = requests.get(f"https://api.yyy001.com/api/whois?domain={domain}")
        response.raise_for_status()
        data = response.json()['data']['whois_info']
        re = f"域名注册商:{data['registrar']}\n日期情况:{data['dates']}\n域名状态:{data['status']}"
        return re
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

def YiYan():
    try:
        response = requests.get("https://api.yyy001.com/api/yiyan?charset=UTF-8")
        response.raise_for_status()
        data = response.text
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

# 起零API

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
    import re
    msg = re.sub(r'(\*\*|__)|(\*|_)|\~\~|\[([^\]]+)\]\(([^)]+)\)|!\[([^\]]*)\]\(([^)]*)\)|`([^`]+)`|```[^`]*```|^#{1,6}\s.*$|^>.*$|^[\-*]\s+.*$', '', msg) # 过滤Markdown语法
    msg = msg.replace("#",'')
    msg = msg.replace('*','')
    msg = msg.replace('- ','')
    msg = re.sub(r'<.*?>', '', msg) # 过滤HTML
    # msg = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', msg) # 过滤URL
    return msg

# 高德API
def GetWeather(city):
    try:
        response = requests.get(f'https://restapi.amap.com/v3/geocode/geo?address={city}&output=json&key={gaode_key}')
        response.raise_for_status()
        data = response.json()
        if data['status'] == '1' and data['info'] == 'OK':
            i = data['geocodes'][0]
            adcode = i['adcode']
            full_name = i['formatted_address']
        else:
            return "获取天气失败，请检查城市名是否正确"
        try:
            response = requests.get(f'https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={gaode_key}')
            response.raise_for_status()
            data = response.json()
            if data['status'] == '1' and data['info'] == 'OK':
                i = data['lives'][0]
                weather = i['weather']
                temperature = i['temperature']
                wind_direction = i['winddirection']
                wind_power = i['windpower']
                return f"\n{full_name}\n天气: {weather}\n温度: {temperature}°C\n风向: {wind_direction}\n风力: {wind_power}\n数据来源: 高德开放平台 上报时间: {i['reporttime']}"
        except:
            return "获取天气失败，请检查城市名是否正确"
        print(data)
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

def GetUserConfig():
    with open('./user.json','r',encoding='utf-8') as f:
        user = json.load(f)
    return user

def SendGroupListMsg(data, msg):
    group_id = data['group_id']
    body = {"group_id": group_id, 'msg_id':data['message_id'], "message": msg}
    try:
        response = requests.post(url=f"{api_ip}/send_group_msg", json=body)
        print(response.json())
        return response.json()
    except requests.RequestException as e:
        print("Error:", e)
        return None
    
def SendGroupTextMsg(data, msg):
    group_id = data['group_id']
    body = {"group_id": group_id, 'message_id':data['message_id'], "message": [{"type": "text", "data": {"text": msg}}]}
    try:
        response = requests.post(url=f"{api_ip}/send_group_msg", json=body)
        print(response.json())
        return response.json()
    except requests.RequestException as e:
        print("Error:", e)
        return None

def SendGroupImg(data,img):
    try:
        SendGroupMsg(data,f"[CQ:image,file={img}]")
    except Exception as e:
        print("Error:", e)
        SendGroupMsg(data,f"获取图片失败{str(e)}")


def SendGroupMsg(data,msg,double=False):
    print("[Info]发送消息: " + str(msg))
    if data['group_id'] not in without_ban:
        msg = BanKeyWord(msg)
    try:
        
        if double:
            requests.get(f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false")
        return requests.get(f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false").text
    except Exception as e:
        print("发送失败:" + f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false")
        return e

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

def WriteUserJson(user):
    with open('./user.json','w',encoding='utf-8') as f:
        json.dump(user,f,ensure_ascii=False,indent=4)
    return

def SingIn(data):
    uid = GetUid(data)
    with open('./user.json','r') as f:
        user = json.load(f)
    datatime = datetime.now().strftime('%Y-%m-%d')
    if user[uid]['lashSign'] == datatime:
        return '今天已经签到过了，请明天再来'
    else:
        user[uid]['lashSign'] = datatime
        user[uid]['SignDays'] = str(int(user[uid]['SignDays'])+1)
        user[uid]['Coin'] = str(int(user[uid]['Coin'])+10)
        threading.Thread(target=WriteUserJson,args=(user,)).start()
        return f"签到成功，获得10金币\n总金币：{AddCoin(data,10)}\n\n已经签到了{user[uid]['SignDays']}天"

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

def getNowTimeStamp():
    # 获取年月日
    year = datetime.now().year
    month = datetime.now().month
    day = datetime.now().day
    return int(year * 10000 + month * 100 + day)

def TodayYunshi(data):
    user_id = data.get('real_user_id')
    # 设置随机种子
    random.seed(str(user_id) + str(getNowTimeStamp()))
    r = random.randint(1,100)
    with open('./yunshi.json','r',encoding='utf-8') as f:
        yunshi = json.load(f)
    max_good = len(yunshi['good'])
    max_bad = len(yunshi['bad'])
    good = []
    bad = []
    for i in range(2):
        t = random.randint(0,max_good-2)
        good.append(yunshi['good'][t])
        yunshi['good'].pop(t)
        max_bad -= 1
    for i in range(2):
        t = random.randint(0,max_bad-1)
        bad.append(yunshi['bad'][t])
        yunshi['bad'].pop(t)
        max_bad -= 1
    #print(good)
    #print(bad)
    # good = [random.choice(yunshi['good']),random.choice(yunshi['good'])]
    # bad = [random.choice(yunshi['bad']),random.choice(yunshi['bad'])]
    if r <= 10:
        return '[CQ:image,file=https://t.alcy.cc/mp]\n§ 大凶 §\n' + '★☆☆☆☆\n' + f"宜:\n  诸事不宜\n忌:\n  {bad[0]}\n  {bad[1]}"
    elif r > 10 and r <= 30:
        return '[CQ:image,file=https://t.alcy.cc/mp]\n§ 小凶 §\n' + '★★☆☆☆\n' + f"宜:\n  {good[0]}\n  {good[1]}\n忌:\n  {bad[0]}\n  {bad[1]}"
    elif r > 30 and r <= 60:
        return '[CQ:image,file=https://t.alcy.cc/mp]\n§ 中平 §\n' + '★★★☆☆\n' + f"宜:\n  {good[0]}\n  {good[1]}\n忌:\n  {bad[0]}\n  {bad[1]}"
    elif r > 60 and r <= 80:
        return '[CQ:image,file=https://t.alcy.cc/mp]\n§ 小吉 §\n' + '★★★★☆\n' + f"宜:\n  {good[0]}\n  {good[1]}\n忌:\n  {bad[0]}\n  {bad[1]}"
    elif r > 80 and r <= 100:
        return '[CQ:image,file=https://t.alcy.cc/mp]\n§ 大吉 §\n' + '★★★★★\n' + f"宜:\n  {good[0]}\n  {good[1]}\n忌:\n  诸事皆宜"

def NazhaPiaofang():
    try:
        data = requests.get('https://60s-api.viki.moe/v2/maoyan').json()['data']['list']
        for i in data:
            if i['movie_name'] == '哪吒之魔童闹海':
                return f"\n{i['movie_name']}实时数据：(来源：猫眼)\n票房:{i['box_office']}({i['box_office_desc']})\n排名:{i['rank']}"
        return '哪吒之魔童闹海未找到'
    except Exception as e:
        return str(e.args)

def TimestampToTime(timestamp):
    #import datetime
    #timestamp = 1557732923
    time_struct = datetime.fromtimestamp(timestamp)
    return time_struct.strftime('%Y-%m-%d')

def GetHistoryToday():
    try:
        data = requests.get('https://v2.xxapi.cn/api/history').json()
        try:
            if data['code'] != 200:
                return '获取失败'
            ret = ""
            for i in data['data']:
                ret += i
            return ret
        except:
            return '获取失败'
    except Exception as e:
        return str(e.args)

# 40code API

def GetWorkInfo(id):
    try:
        data = requests.get(f'https://api.abc.520gxx.com/work/info?id={id}&token=').json()
        if data['code'] != 1:
            return '获取失败'
        data = data.get('data')
        if data['delete'] == 1:
            return '本作品已删除'
        if data['opensource'] == 1:
            opensource = '开源'
        else:
            opensource = '非开源'
        if data['publish'] == 1:
            publish = '已发布'
        else:
            publish = '未发布'
        return f"\n{data['name']}\nID:{data['id']}\n{publish} {opensource}\n作者:{data['nickname']}({data['author']})\n查看:{data['look']} 点赞:{data['like']} 收藏:{data['num_collections']}"
    except Exception as e:
        return f"错误：{e}"

def Get40codeCoinList():
    try:
        data = requests.get('https://api.abc.520gxx.com/user/clist?token=').json()
        if data.get('code') != 1:
            return '获取失败'
        ret = ""
        data = data.get('data')
        for i in data:
            ret += f"\n{i['nickname']}({i['id']}) 金币数：{i['coins']}"
        return ret
    except Exception as e:
        return f"错误：{e}"

def Get40codeUserInfo(id):
    try:
        data = requests.get(f'https://api.abc.520gxx.com/user/info?id={str(id)}&token=').json()
        if data['code'] != 1:
            return '获取失败'
        data = data.get('data')[0]
        #ret = type(data)
        ret = f"\n{data['nickname']}  ID: {data['id']}\n金币数：{data['coins']}\n粉丝数: {data['fan']} 关注数: {data['follow']}\n注册时间: {TimestampToTime(data['signtime'])}\n最后活跃时间: {TimestampToTime(data['last_active'])}"
        return ret
    except Exception as e:
        return f"错误：{e}"

def Get40codeNewestUser():
    try:
        data = requests.post('https://api.abc.520gxx.com/search/?token=',json={"name":"","author":"","type":1,"s":1,"sid":"","fl":0,"fan":0,"follow":0,"page":"1","folder":0}).json()
        if data['code'] != 1:
            return '获取失败'
        data = data.get('data').get('user')
        return data[0]['id']
    except Exception as e:
        return f"错误：{e}"

def build_markdown_segment_text(content: str, keyboard_template: dict) -> dict:
    """
    构建文本形式的 Markdown Segment (Base64 编码)
    
    :param content: 消息文本内容
    :param keyboard_template: 已审核的按钮模板
    :return: OneBot 兼容的 MessageSegment
    """
    # 构建 Gensokyo Markdown 结构
    markdown_data = {
        "markdown": content,
        "keyboard": keyboard_template
    }
    
    # 转换为 JSON 并 Base64 编码
    json_str = json.dumps(markdown_data, ensure_ascii=False)
    base64_str = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    
    return {
        "type": "markdown",
        "data": {
            "data": f"base64://{base64_str}"
        }
    }        

def GetMemAll():
    mem = psutil.virtual_memory()
    return round(mem.total / (1024.0 ** 3),2)

@app.route("/",methods=["POST","GET"])
def root():
    data = request.json
    if 'group_id' not in data:
        msg = str(data['raw_message'])
        if msg == '时间':
            SendPrivateMsg(data,f"当前时间是{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return {}
        SendPrivateMsg(data,"命令不正确，目前私聊只支持/时间指令")
    # print(data)
    msg = str(data['raw_message'])
    if msg == 'help':
        SendGroupMsg(data,help_text)
        return 'Successfully',200
    
    if msg == '时间':
        SendGroupMsg(data,f"当前时间是{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 'Successfully',200
    
    elif msg.startswith("审核 "):
        SendGroupMsg(data,f"已添加白名单,祝您游戏愉快")
        msg = msg.replace("审核 ","")
        with Client(rcon_host, rcon_port, passwd=rcon_password) as c:
            c.run('whitelist add ' + msg)
            c.run('whitelist save')
            return 'Successfully',200
    
    elif msg == '40code':
        SendGroupMsg(data,f"检测结果: 40code正常运行")
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
    
    if msg == '随机图':
        SendGroupMsg(data,"正在获取随机图")
        SendGroupImg(data,"https://t.alcy.cc/ycy")
        return 'Successfully',200

    if msg.startswith("随机图 "):
        msg = msg.replace("随机图 ","")
        SendGroupMsg(data,"正在获取随机图(没有发出来就是被和谐了)")
        if msg == '原神':
            SendGroupImg(data,"https://t.alcy.cc/ys")
        elif msg == 'AI':
            SendGroupImg(data,"https://t.alcy.cc/ai")
        # https://t.alcy.cc/mp
        elif msg == '竖版':
            SendGroupImg(data,"https://t.alcy.cc/mp")
        elif msg == '萌版':
            SendGroupImg(data,"https://t.alcy.cc/moe")
        elif msg == "白色背景":
            SendGroupImg(data,"https://t.alcy.cc/bd")
        elif msg == "原神竖版":
            SendGroupImg(data,"https://t.alcy.cc/ysmp")
        elif msg == "七瀨胡桃":
            SendGroupImg(data,"https://t.alcy.cc/lai")
        elif msg == "头像":
            SendGroupImg(data,"https://t.alcy.cc/tx")
        elif msg == "乐青源":
            SendGroupImg(data,"https://api.imlazy.ink/img")
        elif msg == "樱花":
            SendGroupImg(data,"https://www.dmoe.cc/random.php")
        elif msg == "东方":
            SendGroupImg(data,"https://img.paulzzh.com/touhou/random")
        elif msg == "R18":
            time.sleep(10)
            SendGroupMsg(data,'拒绝黄色，从你做起😅')
        elif msg == "风景":
            SendGroupImg(data,"https://picsum.photos/580/300")
        elif msg == "三次元":
            SendGroupImg(data,"https://v2.xxapi.cn/api/meinvpic?return=302")
        elif msg == "原神2":
            SendGroupImg(data,"https://api.suyanw.cn/api/ys")
        elif msg == "甘城猫猫":
            SendGroupImg(data,"https://api.suyanw.cn/api/mao")
        elif msg == "碧蓝航线":
            SendGroupImg(data,"https://image.anosu.top/pixiv/direct?r18=0&keyword=azurlane")
        else:
            SendGroupImg(data,'https://i.pixiv.re/img-original/img/2022/10/28/00/00/11/102280854_p0.png')
            SendGroupMsg(data,"开始从pixiv中获取图片")
            ret = requests.get(f"https://image.anosu.top/pixiv/json?keyword={msg}").json()[0]
            img = ret['url']
            SendGroupImg(data,img)
        return 'Successfully',200
    if msg.startswith("绑定QQ号 "):
        msg = msg.replace("绑定QQ号 ","")
        return 'Successfully',200
    
    if msg == '切换模型':
        SendGroupMsg(data,'可是你并没有指定模型名称....')
        return 'Successfully',200

    if msg == '测试':
        SendGroupMsg(data, f'40code[CQ:image,file=https://t.alcy.cc/mp]')
        return 'Successfully',200
    
    if msg == "40code 金币榜" or msg == "40code coin":
        SendGroupMsg(data,Get40codeCoinList())
        return 'Successfully',200

    if msg.startswith('40code user '):
        msg = msg.replace('40code user ', '')
        SendGroupMsg(data,Get40codeUserInfo(msg))
        return 'Successfully',200

    if msg.startswith('40code work '):
        msg = msg.replace('40code work ', '')
        SendGroupMsg(data,GetWorkInfo(msg))
        return 'Successfully',200

    if msg == '一言':
        SendGroupMsg(data,f"{YiYan()}")
        return 'Successfully',200
    
    if msg == '今日运势':
        SendGroupMsg(data, TodayYunshi(data=data))
        return 'Successfully',200
    
    if msg.startswith('域名查询 '):
        msg = msg.replace("域名查询 ","")
        msg = msg.replace(" ",".")
        domain = msg
        SendGroupMsg(data,'正在查询')
        SendGroupMsg(data,str(DomainInfo(domain)))
        return 'Successfully',200
    
    if msg == '获取ID':
        SendGroupMsg(data,f"用户ID: {data['real_user_id']}")
        return 'Successfully',200
    
    if msg == 'today':
        SendGroupMsg(data,f"正在查询历史上的今天")
        print('[Debug]收到:today')
        SendGroupMsg(data,GetHistoryToday())
        print('[Debug]发送:today')
        return 'Successfully',200
    
    if msg.startswith("status "):
        msg = msg.replace("status ","")
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
        
    elif msg == 'check gitlab':
        t = requests.get('https://git.dev.scerpark.cn/-/liveness?token=M7XVNuCsyxM9WifxQ_ff').json().get('status','error')
        if t == 'ok':
            SendGroupMsg(data,"Gitlab状态正常")
        else:
            SendGroupMsg(data,"Gitlab状态异常")
        return 'Successfully',200
    elif msg.startswith("mc "):
        SendGroupMsg(data,"该指令已禁用")
        return 'Successfully',200
    
    elif msg == '本次会话':
        SendGroupMsg(data,f"\nGroup_ID={data['group_id']}\nReal_Group_ID={data['real_group_id']}\nUser_ID={data['user_id']}\nMessage_ID={data['message_id']}\nRaw_Message={data['raw_message']}\nMessage_Type={data['message_type']}\nSub_Type={data['sub_type']}\nFont={data['font']}\nSelf_Id={data['self_id']}\nPost_Type={data['post_type']}\n--End--")
        return 'Successfully',200
    
    elif msg.startswith('echo '):
        SendGroupMsg(data,msg.replace('echo ','',1),True)
        return 'Successfully',200
    
    elif msg.startswith("切换模型 "):
        msg = msg.replace("切换模型 ","")
        # if msg in deepseek_model:
        #     SendGroupMsg(data,f"是抖M吗你")
        if msg not in model_list and msg not in breath_all_model and msg != 'grok-3-mini-devx':
            SendGroupMsg(data,f"模型不存在")
            return 'Successfully',200
        uid = GetUid(data)
        user = GetUserConfig()
        user[uid]['model'] = msg
        threading.Thread(target=WriteUserJson,args=(user,)).start()
        SendGroupMsg(data,f"已将模型切换为{msg}")
        return 'Successfully',200
        
    elif msg.startswith("bilibili "):
        msg = msg.replace("bilibili ","")
        if msg.startswith('search '):
            msg = msg.replace('search ','')
            if msg == '':
                SendGroupMsg(data,f"请输入搜索内容")
                return 'Successfully',200
            # SendGroupMsg(data,f"正在搜索...")
            ret = BilibiliSearch(msg)
            SendGroupMsg(data,ret)
            return 'Successfully',200
    
    elif msg.startswith('weather '):
        msg = msg.replace('weather ','')
        # SendGroupMsg(data,f"正在查询...")
        SendGroupMsg(data,GetWeather(msg))
        return 'Successfully',200

    elif msg == '当前模型':
        SendGroupMsg(data,f"当前模型为{GetModel(data)}")
        return 'Successfully',200
    
    elif msg == '抖音热榜':
        SendGroupMsg(data,DouyinHot())
        return {}
    
    elif msg == '哪咤票房' or msg == '哪吒票房':
        SendGroupMsg(data,NazhaPiaofang())
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
        retu = '\n'
        for i in model_list:
            retu += f"{i}\n"
        retu += f"共{len(model_list)}个模型可用"
        retu += '\n非常感谢灵息AI提供的大量模型'
        SendGroupMsg(data,retu)
        return {}
    
    elif msg == '模型列表 通义千问':
        # SendGroupMsg(data,f"模型列表:")
        retu = '\n'
        for i in qwen_3_model:
            retu += f"{i}\n"
        retu += f"共{len(qwen_3_model)}个模型可用"
        SendGroupMsg(data,retu)
        return {}
    
    elif msg == '模型列表 详细' or msg == "模型列表 详细 1":
        # SendGroupMsg(data,f"模型列表:")
        retu = '\n§ 以下模型均由灵息AI提供 §\n'
        for i in breath_all_model[:20]:
            retu += f"{i}\n"
        retu += f"列表过长, 请输入 模型列表 详细 2 查看更多"
        SendGroupMsg(data,retu)
        return {}
    
    elif msg == '模型列表 详细 2':
        # SendGroupMsg(data,f"模型列表:")
        retu = '\n§ 以下模型均由灵息AI提供 §\n'
        for i in breath_all_model[20:]:
            retu += f"{i}\n"
        retu += f"共{len(breath_all_model[20:])}个模型可用"
        SendGroupMsg(data,retu)
        return {}
    
    elif msg == '40code 用户数':
        SendGroupMsg(data,f"{Get40codeNewestUser()}")
        return 'Successfully',200

    elif msg == 'zerocat 用户数':
        SendGroupMsg(data,f"{GetZeroCatUser()}")
        return 'Successfully',200

    elif msg == '小盒子 用户数':
        SendGroupMsg(data,f"{GetBoxUser()}")
        return 'Successfully',200


    elif msg.startswith('bing search '):
        # SendGroupMsg(data,f"正在搜索...")
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
        #SendGroupMsg(data,'本功能暂时禁用')
        SendGroupMsg(data,SingIn(data))
        return {}
    
    elif msg == '商店':
        SendGroupMsg(data,'商店为空')
        return {}
    
    elif msg == '关于':
        SendGroupMsg(data,TestMarkdown())
        return {}

    # 判断是否为流式对话
    if data['group_id'] in StreamInfo and data['user_id'] in StreamInfo[data['group_id']]:
        SendGroupMsg(data,'正在思考(当前处于流式对话模型)')
        '''流式对话'''
        ret = StreamChat(data)
        SendGroupMsg(data,ret)
        SendGroupMsg(data,'对话完成')
        return {}
    
    SendGroupMsg(data,f"正在思考...(如果没有内容发出，就是被和谐了)")
    print("[Debug]Call Model: " + GetModel(data))
    SendGroupMsg(data,NoStreamChat(GetModel(data),msg))
    return {}

import json

def TestMarkdown():
    t = {
        "keyboard": {
            "id":"102646446_1746274607"
        }
    }
    # 将字典转换为 JSON 字符串
    t_json = json.dumps(t, ensure_ascii=False)
    # 对 JSON 字符串进行 Base64 编码
    encoded_t = encode_to_base64(t_json)
    # 构建最终的回复字符串
    reply = f"[CQ:markdown,data=base64://{encoded_t}]"
    return reply

if __name__ == "__main__":
    if not os.path.exists('./temp'):
        os.makedirs('./temp')
    else:
        shutil.rmtree('./temp')
        os.makedirs('./temp')
    with open('./whitelist.json', 'r') as f:
        whitelist = json.load(f)
    # TestMarkdown()
    if not os.path.exists('./user.json'):
        with open('./user.json', 'w') as f:
            json.dump({}, f)
    app.run(host='0.0.0.0', port=8090, debug=True, threaded=True)
