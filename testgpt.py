import time
import json
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI

# 配置OpenAI客户端（保持不变）
gpt_model = "gpt-4o"
OPENAI_API_KEY = "sk-gLJpO9I10cMfjn0PYz80SSwELl84fmTyKjhYlMUwkyANTfpf"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1")


def local_image_to_base64(image_path, image_format="JPEG"):
    try:
        with Image.open(image_path) as img:
            if image_format == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            buffer = BytesIO()
            img.save(buffer, format=image_format, quality=90)
            buffer.seek(0)

            # 编码为Base64字符串，并添加API要求的前缀
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/{image_format.lower()};base64,{base64_str}"

    except Exception as e:
        print(f"本地图片处理失败：{str(e)}")
        return None


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def build_multimodal_content(text, image_data_list=None):
    content = [{"type": "text", "text": text}]

    if image_data_list:
        for img_data in image_data_list:
            # 判断是Base64字符串（本地图片）还是在线URL
            if img_data.startswith("data:image/"):
                content.append({"type": "image_url", "image_url": {"url": img_data}})
            else:
                content.append({"type": "image_url", "image_url": {"url": img_data}})

    return content


# -------------------------- 核心改动1：增强GPT调用函数，强制JSON输出并校验 --------------------------
def gpt_chat(content, conversation, max_retries=3):
    """发送聊天请求（支持本地图片的Base64编码，强制输出JSON并校验）"""
    # 1. 在用户请求末尾添加“强制JSON输出”的约束（不修改原提示词核心）
    if isinstance(content, list):
        # 若为多模态内容（文本+图片），找到文本部分并追加约束
        for item in content:
            if item["type"] == "text":
                item[
                    "text"] += "\n\n### 输出要求\n必须以纯JSON格式输出结果，不要包含任何非JSON内容（如解释性文字、代码块标记）。JSON结构需符合标准语法，键名使用双引号，值为字符串或数字类型。根据任务需求定义合理的JSON字段（如round1_result、red_region_positions、classification_reason、error_assessment等），确保字段与任务目标对应。"
                break
        user_message = {"role": "user", "content": content}
    else:
        # 若为纯文本内容，直接追加约束
        content += "\n\n### 输出要求\n必须以纯JSON格式输出结果，不要包含任何非JSON内容（如解释性文字、代码块标记）。JSON结构需符合标准语法，键名使用双引号，值为字符串或数字类型。根据任务需求定义合理的JSON字段（如round1_result、red_region_positions、classification_reason、error_assessment等），确保字段与任务目标对应。"
        user_message = {"role": "user", "content": content}

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                temperature=0.2,  # 低温度确保输出稳定，减少格式错乱
                messages=conversation + [user_message],
                stream=False
            )
            # 2. 提取模型输出并尝试解析JSON（校验格式正确性）
            raw_output = response.choices[0].message.content.strip()
            # 移除可能的代码块标记（如```json、```）
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()  # 截取JSON内容，去除首尾标记
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()

            # 3. 解析JSON并返回（若解析失败则重试）
            json_output = json.loads(raw_output)
            return json_output

        except json.JSONDecodeError as e:
            error_msg = f"JSON格式错误：{str(e)[:200]}"
            print(f"API返回内容非标准JSON (尝试 {retry_count + 1}/{max_retries}): {error_msg}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)
        except Exception as e:
            error_msg = str(e)[:250]
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {error_msg}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)

    print("已达到最大重试次数，请求失败。返回空JSON。")
    return {"error": "max_retries_reached", "message": "未能获取有效JSON输出"}
def gpt_chat1(content, conversation, max_retries=3):
    """发送聊天请求（支持本地图片的Base64编码）"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            if isinstance(content, list):
                user_message = {"role": "user", "content": content}
            else:
                user_message = {"role": "user", "content": content}

            response = client.chat.completions.create(
                model=gpt_model,
                temperature=0.2,
                messages=conversation + [user_message],
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)[:250]
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {error_msg}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)
    print("已达到最大重试次数，请求失败。")
    return None

# -------------------------- 核心改动2：修改多轮对话函数，接收JSON输出并存储 --------------------------
def round_1(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    # 调用增强后的gpt_chat，直接获取JSON输出
    round1_answer_json = gpt_chat(content, conversation)
    # 将JSON转为字符串存入对话历史（确保后续对话能参考）
    round1_answer_str = json.dumps(round1_answer_json, ensure_ascii=False, indent=2)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round1_answer_str})
    return conversation, round1_answer_json  # 返回对话历史+JSON结果（方便后续使用）


def round_2(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round2_answer_json = gpt_chat(content, conversation)
    round2_answer_str = json.dumps(round2_answer_json, ensure_ascii=False, indent=2)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round2_answer_str})
    return conversation, round2_answer_json

def round_3(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round3_answer = gpt_chat1(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round3_answer})
    return conversation


def round_4(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round4_answer_json = gpt_chat(content, conversation)
    round4_answer_str = json.dumps(round4_answer_json, ensure_ascii=False, indent=2)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round4_answer_str})
    return conversation, round4_answer_json


def round_5(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round5_answer_json = gpt_chat(content, conversation)
    round5_answer_str = json.dumps(round5_answer_json, ensure_ascii=False, indent=2)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round5_answer_str})
    return conversation, round5_answer_json


# -------------------------- 核心改动3：主函数接收JSON结果并可选保存到文件 --------------------------
if __name__ == "__main__":
    # 1. 本地图片路径（保持不变）
    local_image_paths1 = [
        r"C:\Users\34517\Desktop\zuhui\Time_is_not_Enough-main\spectrogram_plots\computer_class_1\computer_class_1_sample_153.png",
    ]
    local_image_paths2 = [
        r"C:\Users\34517\Desktop\zuhui\Time_is_not_Enough-main\line_plots\computer_class_1\computer_class_1_sample_153.png",

    ]
    local_image_paths3=[

    ]

    # 2. 图片转为Base64（保持不变）
    image_data_list1 = []
    image_data_list2 = []
    image_data_list3 = []

    for img_path in local_image_paths1:
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")
        if base64_str:
            image_data_list1.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")

    for img_path in local_image_paths2:
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")
        if base64_str:
            image_data_list2.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")

    for img_path in local_image_paths3:
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")
        if base64_str:
            image_data_list3.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")
    # 3. 多轮对话（接收JSON结果并打印/保存）
    conversation = []
    all_results = {}  # 存储所有轮次的JSON结果


    knowledge = ("It is derived from a government-funded study called \"Powering the Nation\". "
              "This study aims to collect data on consumers' electricity usage behavior in households to help reduce carbon emissions in the UK.\n"
              "The dataset includes electricity consumption records of 251 households, with a sampling interval of two minutes, and the data was continuously collected for one month. "
               "Among them, the length of each time-series data is 720 (that is, one reading is collected every two minutes within 24 hours, calculated as 24×60÷2=720). "
               "The categories in the dataset are divided into two types: desktop computers and laptop computers, which are presumably used to distinguish the electricity consumption characteristics of these two types of devices.\n"
                 )


    print("=== 第一轮对话 ===")
    first_prompt = ("You are a Time Series Classification and Model Interpretability Expert.I have a long time series of data for a specific category. "
                    "I have converted this data into a time-frequency representation using the Short-Time Fourier Transform (STFT). "
                    "I would like you to help me identify the features present in the time series data. I will provide you with background information about the data charts\n"
                    "- The horizontal axis represents the time axis, and the width of each pixel is a time window.\n"
                    "- The vertical axis represents the frequency axis, which contains the frequency components within a time window.\n"
                    "- The pixel value represents the signal energy, which is calculated from the complex matrix output by the short-time Fourier transform. "
                    "Each element of the complex matrix is a complex number, containing amplitude and phase. The magnitude of the complex number is the energy.\n\n"
                    "<Task>: I will provide you with time-frequency maps corresponding to samples of a certain category in the dataset. "
                    "You need to extract valuable information from these segments to help me classify them in the subsequent time-series classification task.\n\n"
                    "<Target>: Extract valuable information from this time-frequency image for subsequent time series classification tasks. "
                    "The output should include structured information in  JSON format.\n"
                    "- **Overall analysis of time-frequency diagram**：Attempt to describe all the overall characteristic information of this sample's time-frequency diagram. "
                    "The description must cover at least the following aspects:\n"
                    "- Frequency distribution range: Which frequency ranges does the energy mainly concentrate in, and is there an obvious dominant frequency.\n"
                    "- Time stability: Whether the frequency components change smoothly over time, and whether there are sudden frequency components appearing.\n"
                    "- Energy distribution pattern: Whether the overall energy level is uniform, and whether the color intensity changes follow the expected periodicity.\n"
                    "- Periodicity: Whether a repeating stripe pattern can be observed on the time-frequency diagram, and the inclination angle of the stripes reflects the frequency change speed.\n"
                    "- **Local analysis of time-frequency diagramm**：Attempt to describe all the local characteristic information of this sample's time-frequency diagram. "
                    "The description must cover at least the following aspects:\n- What time zones and frequency ranges are the high-energy regions distributed in\n"
                    "- Whether there is a short-term high-frequency energy burst (local bright spot), if there is a corresponding time and frequency range, what is it\n"
                    "- Whether there are abnormal frequency spectra caused by truncation at the beginning and end (usually manifested as high-frequency energy at the edges)")

    conversation, round1_json = round_1(conversation, first_prompt, image_data_list1)
    all_results["round1"] = round1_json  # 存入总结果字典
    print("第一轮JSON输出：")
    print(json.dumps(round1_json, ensure_ascii=False, indent=2))  # 格式化打印
    print("-" * 80)

    print("\n=== 第二轮对话 ===")
    second_prompt = ("You are a Time Series Classification and Model Interpretability Expert. I have a long time-series dataset of a specific category. "
                     "I have converted this data into a graph and need your help to identify the features in the time-series data. "
                     "I will provide you with background information about the data graph:\n"
                     "The horizontal axis represents the time-series index.\n"
                     "The vertical axis represents the values of the time-series data.\n"
                     "<Task>: I will provide you with graphs corresponding to samples of a specific category in this dataset. "
                     "You need to extract valuable information from these segments to assist me in the subsequent time-series classification task.\n"
                     "<Target>: Please help me extract valuable information from this line graph for use in the subsequent time-series classification task. "
                     "The output must include structured information and be presented in JSON format:\n"
                     "analysis of line graph : Attempt to describe all the characteristic information of this sample graph. "
                     "The description must cover at least the following aspects: periodicity, stability, trends, peaks, valleys, other important features, and their corresponding time (intervals.")

    conversation, round2_json = round_2(conversation, second_prompt, image_data_list2)
    all_results["round2"] = round2_json
    print("第二轮JSON输出：")
    print(json.dumps(round2_json, ensure_ascii=False, indent=2))
    print("-" * 80)

    # （可选）将所有轮次结果保存到本地JSON文件
    with open("gpt_multimodal_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("所有轮次结果已保存到：gpt_multimodal_results.json")

    category = ("the second category (laptop computers)")

    print("\n=== 第三轮对话 ===")
    third_prompt = ("You are a Time Series Classification and Model Interpretability Expert. I have a long time-series dataset of a specific category. "
                    "The background of this time-series data is as follows:\n"
                    +knowledge+
                    "I have converted this data into graphs and time-frequency diagrams, and I will provide you with the text-based feature analysis and description of them. At the same time, "
                    "I will provide you with a heatmap that reflects the decision basis of the black-box model. "
                    "It is known that the black-box model classifies the above sample time-series data into "+ category +". "
                    "I need you to combine all the information and conduct reasoning to help me judge whether the model's decision result is correct. "
                    "If it is correct, analyze why the model focuses on the areas reflected in the heatmap; if it is incorrect, analyze which areas should be focused on instead.")
    conversation = round_3(conversation, third_prompt, image_data_list3)
    print("第san轮回答：")
    print(conversation[-1]["content"])
    print("-" * 80)