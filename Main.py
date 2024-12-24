from rcon import Client # type: ignore

# RCON服务器地址和端口
rcon_host = '127.0.0.1'  # 替换为你的服务器地址
rcon_port = 25575
rcon_password = '@Fkchh000'

# 创建RCON客户端
with Client(rcon_host, rcon_port, passwd=rcon_password) as client:
    # 发送命令到服务器
    response = client.run('list')  # 你可以替换为任何你想要执行的命令
    print(response)