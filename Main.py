from rcon import Client # type: ignore

# RCON服务器地址和端口
rcon_host = '127.0.0.1'
rcon_port = 25575
rcon_password = '@Fkchh000'

def Run(command,c):
    response = c.run(command)
    return response

if __name__ == '__main__':
    with Client(rcon_host, rcon_port, passwd=rcon_password) as c:
        while True:
            try:
                print(Run(input('请输入命令：'), c))
            except KeyboardInterrupt:
                print('\n退出')
                exit()