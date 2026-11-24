from tensorboard import program
import config

# 创建 TensorBoard 对象
tb = program.TensorBoard()

# 配置日志目录
# argv 是 Argument Vector（参数向量）的缩写，用于接收命令行参数列表。
#假设在终端中执行  tensorboard --logdir ./logs，对应的参数列表就是    ['tensorboard', '--logdir', './logs']
# 命令行参数列表的第一个元素通常是程序名称，故用None作占位符，后面的元素才是真正需要解析的参数。前面创建TensorBoard对象相当于已经指定了程序名称
tb.configure(argv=[None, '--logdir', str(config.LOGS_DIR)])

# 启动 TensorBoard 服务
url = tb.launch()

print(url)

# 防止程序退出导致服务停止
input('按回车键关闭 TensorBoard')