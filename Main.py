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

app = Flask(__name__)

help_text = """/mc [指令] 执行服务器指令
/help 获取帮助
/status [条目] 查询服务器状态
 可用条目：
    运行内存：查询服务器总内存
    CPU：查询服务器CPU占用率
    运行时长：查询服务器运行时长
/审核 [玩家名] 将玩家加入白名单"""
rcon_host = '127.0.0.1'
rcon_port = 25575
rcon_password = '@Fkchh000'
api_ip = 'http://127.0.0.1:3000'
whitelist = {}


def GetMemberRole(data):
    id = data['user_id']

def SendGroupMsg(data,msg):
    try:
        requests.get(f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false")
    except:
        print("发送失败:" + f"{api_ip}/send_group_msg?group_id={data['group_id']}&msg_id={data['message_id']}&message={msg}&auto_escape=false")

def SendPrivateMsg(data,msg):
    try:
        requests.get(f"{api_ip}/send_private_msg?user_id={data['user_id']}&message={msg}&auto_escape=false")
    except:
        print("发送失败:" + f"{api_ip}/send_private_msg?user_id={data['user_id']}&message={msg}&auto_escape=false")

def get_cpu_usage():
    """ 获取当前系统的CPU使用率 """
    cpu_percent = psutil.cpu_percent(interval=1)
    return cpu_percent

def GetMemAll():
    mem = psutil.virtual_memory()
    return round(mem.total / (1024.0 ** 3),2)

@app.route("/",methods=["POST","GET"])
def root():
    data = request.json
    if 'group_id' not in data:
        SendPrivateMsg(data,"请使用群聊发送命令")
    print(data)
    msg = str(data['raw_message'])
    if msg.startswith("/status "):
        msg = msg.replace("/status ","")
        if msg == "运行内存":
            mem = GetMemAll()
            SendGroupMsg(data,f"总内存为{mem}GB")
        elif msg == "CPU":
            cpu = get_cpu_usage()
            SendGroupMsg(data,f"CPU占用为{cpu}%")
        elif msg == "运行时长":
            # 获取系统启动时间
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            now = datetime.now()
            uptime = now - boot_time
            SendGroupMsg(data,f"系统运行时长为{uptime}")
        else:
            SendGroupMsg(data,"请输入正确的参数")
    elif msg.startswith("/mc "):
        # 去掉开头的/mc
        msg = msg.replace("/mc ","")
        with Client(rcon_host, rcon_port, passwd=rcon_password) as c:
            response = c.run(msg)
            SendGroupMsg(data,response)
    elif msg == "/help":
        SendGroupMsg(data,help_text)
    elif msg.startswith("/审核 "):
        # 获取发送者QQ号
        user_id = data["user_id"]
        if user_id not in whitelist:
            msg = msg.replace("/审核 ","")
            with Client(rcon_host, rcon_port, passwd=rcon_password) as c:
                response = c.run('whitelist add ' + msg)
                c.run('whitelist save')
                whitelist[user_id] = msg
                with open('whitelist.json', 'w'):
                    json.dump(whitelist, open('whitelist.json', 'w'))
                SendGroupMsg(data,f'[CQ:at,qq={user_id}] 已添加白名单')
        else:
            SendGroupMsg(data,f'[CQ:at,qq={user_id}] 一个QQ号只能绑定一个MC用户名')
    return {}

if __name__ == "__main__":
    with open('./whitelist.json', 'r') as f:
        whitelist = json.load(f)
    app.run(host='0.0.0.0', port=8090, debug=True, threaded=True)