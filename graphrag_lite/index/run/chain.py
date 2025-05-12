
# 将operations串联成chain

from langchain_core.runnables import RunnableLambda

# define chunk text runnable lambda

# 用langchain实现异步chain，有一系列async operation （op1, op2, op3, op4），有自定义的runnableconfig，有自定义的store，chain里的runnable开始要从store取相关数据，然后再将结果存到store里，同时需要自定义async callback 负责处理progress error等logging信息，给出你的设计