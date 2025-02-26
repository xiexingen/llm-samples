import os
from openai  import OpenAI
import json
from rich import print
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

# 定义函数方法
def check_order_status(order_id):
  """ 查询给定订单的状态(Mock数据) """
  return json.dumps({
    "status": "运输中",
    "order_id": order_id,
  })

def run_conversation():
  # 第1步：定义提示词
  messages = [
    {
      "role": "user",
      "content": "我的订单 WT20250220 是什么状态"
    }
  ]
  # 把函数当成工具集合
  tools = [
    {
      # 定义工具的类型是函数
      "type": "function",
      "function":{
        # 函数的名字，需要和定义好的python函数名一致
        "name": "check_order_status",
        # 这里的描述告诉 deepseek 这个函数的功能
        "description": "查询给定订单的状态，它的入参是 order_id 也就是订单id",
        # 函数的输入参数
        "parameters":{
          "type": "object",
          "properties":{
            "order_id":{
              "type": "string",
              "description": "订单id，如： WT2000555",
            }
          },
          # 参数必填项
          "required":["order_id"]
        }
      }
    }
  ]

  # 第2步：调用 deepseek 的接口
  client = OpenAI(api_key=api_key,base_url="https://api.deepseek.com")
  response = client.chat.completions.create(
    model = "deepseek-chat", # 模型名称
    messages = messages, # 用户提出的问题
    tools= tools, # 函数的定义
    tool_choice="auto",
  )
  response_message = response.choices[0].message
  print(response_message)
  #第一次请求大模型，得到需要调用函数的信息
  #大模型在得知用户请求的时候， 发现这个事情我不知道， 但是我知道谁知道。 那个知道的“人”就是 tool里面定义的函数
  #从返回的结果中生成需要调用函数的相关信息， 函数名和参数
  tool_calls = response_message.tool_calls
  # 第3步：检查模型是否希望调用函数
  if tool_calls:
    available_functions = {
      "check_order_status": check_order_status,
      # 其他函数... 比如查询库存的函数。。。。
    }
    messages.append(response_message)  # 扩展会话内容
    # 第4步：为每个函数调用发送信息和函数响应给模型
    for tool_call in tool_calls:
      function_name = tool_call.function.name
      #获取调用函数
      function_to_call = available_functions[function_name]
      function_args = json.loads(tool_call.function.arguments)
      function_response = function_to_call(
        order_id=function_args.get("order_id"),
      )
      #对字符串进行追加
      messages.append(
        {
          "tool_call_id": tool_call.id,
          "role": "tool",
          "name": function_name,
          "content": function_response,
        }
      )

    # 第4步：从模型获取新的响应，模型此时可以看到函数响应
    # 再次调用， 需要deepseek 帮助我们润色 返回的信息。 函数返回的订单状态和用户的请求，相结合的信息。
    second_response = client.chat.completions.create(
      model="deepseek-chat",
      messages=messages,
    )
    return second_response


# 执行
orderResponse = run_conversation()
print(orderResponse.choices[0].message)