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
                # Base64格式：直接使用
                content.append({"type": "image_url", "image_url": {"url": img_data}})
            else:
                # 在线URL格式：按原逻辑处理
                content.append({"type": "image_url", "image_url": {"url": img_data}})

    return content


def gpt_chat(content, conversation, max_retries=3):
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


# -------------------------- 多轮对话函数（保持不变） --------------------------
def round_1(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round1_answer = gpt_chat(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round1_answer})
    return conversation


def round_2(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round2_answer = gpt_chat(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round2_answer})
    return conversation


def round_3(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round3_answer = gpt_chat(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round3_answer})
    return conversation
def round_4(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round3_answer = gpt_chat(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round3_answer})
    return conversation
def round_5(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round3_answer = gpt_chat(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round3_answer})
    return conversation
# -------------------------- 主函数（使用本地图片路径） --------------------------
if __name__ == "__main__":
    # 1. 替换为你的本地图片路径（支持JPG/PNG）
    local_image_paths1 = [

        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\heatmap-label2.png",
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\stft.png"
    ]
    local_image_paths2 = [
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\heatmap-label2-1.png",

        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot1(1).png",

        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot2(1).png",
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot3(1).png",
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot4(1).png",
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot5(1).png",
    ]
    local_image_paths3 = [
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot3.png"
    ]
    local_image_paths4 = [
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot4.png"
    ]
    local_image_paths5 = [
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot1.png",

        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot2.png",
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot3.png",
        r"C:\Users\34517\Documents\xwechat_files\wxid_8c3i3fb6j5sr22_56e0\msg\file\2025-09\plot\plot\plot4.png",

    ]

    # 2. 将本地图片转换为Base64编码（API可识别）
    image_data_list1 = []
    image_data_list2 = []

    for img_path in local_image_paths1:
        # 若图片是PNG，第二个参数传"PNG"；JPG/JPEG传"JPEG"
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")

        if base64_str:
            image_data_list1.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")
    for img_path in local_image_paths2:
        # 若图片是PNG，第二个参数传"PNG"；JPG/JPEG传"JPEG"
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")

        if base64_str:
            image_data_list2.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")
    # 3. 多轮对话（传入Base64编码的本地图片）
    conversation = []

    print("=== 第一轮对话 ===")
    first_prompt = "You are a Time Series Classification and Model Interpretability Expert. Your core role is to analyze 1D time series data derived from 2D shapes, interpret the decision-making logic of black-box classification models, and evaluate the reliability of model-generated important regions. You should base all reasoning on the provided dataset background and visual information, and answer questions with clear logic and professional terminology.\n#### 1. Dataset Background Knowledge\nThe data are pseudo time series, which are generated by converting 2D shapes into 1D time series. There are a total of 5 classes corresponding to distinct 2D shapes, as follows: - Class 1: Arrowhead; - Class 2: Butterfly; - Class 3: Fish; - Class 4: Seashell; - Class 5: Shield. \n#### 2. Provided Visual Information\nI have input an image to you. This image contains two types of plots: a line plot and a time-frequency plot. The line plot is obtained by converting 1D time series data into 2D visualization; its x-axis represents the time steps of the time series, and the y-axis represents the values of the time series at corresponding time steps. The time-frequency plot reflects the distribution of the time series' frequency components across different time steps; its x-axis is time steps, y-axis is frequency, and the color intensity represents the energy or amplitude of the corresponding time-frequency component. Both plots contain red-marked regions: in the line plot, the red regions correspond to specific time steps , while in the time-frequency plot, the red boxes select specific time-frequency combinations. These red regions are identified using an interpretability method applied to a black-box classification model. According to the model’s output, these red-marked regions (including time steps in the line plot and time-frequency components in the time-frequency plot) are the decision-critical regions that the model relies on to classify the input time series.\n#### 3. Tasks to Perform\nFlocate the positions of each red regions. Then explain why these red-marked regions (time steps in the line plot and time-frequency components in the time-frequency plot) would be the critical regions for the black-box model to classify the input time series into Class 2 (Butterfly) or Class 3 (Fish). Your explanation should connect the characteristics of the red regions to the potential 2D shape features of Class 2 (Butterfly) or Class 3 (Fish) that the time series represents. "
    conversation = round_1(conversation, first_prompt, image_data_list1)
    print("第一轮回答：")
    print(conversation[-1]["content"])
    print("-" * 80)

    print("\n=== 第二轮对话 ===")
    second_prompt = "You are a Time Series Classification and Model Interpretability Expert. Your core role is to analyze 1D time series data derived from 2D shapes, interpret the decision-making logic of black-box classification models, and evaluate the reliability of model-generated important regions. You should base all reasoning on the provided dataset background, previous conversation context, and visual information, and answer questions with clear logic and professional terminology.\n#### 1. Dataset Background Knowledge\nThe data are pseudo time series, which are generated by converting 2D shapes into 1D time series. There are a total of 5 classes corresponding to distinct 2D shapes, as follows: - Class 1: Arrowhead; - Class 2: Butterfly; - Class 3: Fish; - Class 4: Seashell; - Class 5: Shield. Two datasets are constructed from these pseudo time series. \n#### 2. Previous Conversation Context\nIn the last round of conversation, I provided you with an image of a line plot. This line plot was obtained by converting 1D time series data into 2D visualization, with the x-axis representing the time steps of the time series and the y-axis representing the values of the time series at corresponding time steps. The image also contained red-marked time regions, which were identified via an interpretability method applied to a black-box classification model—these red regions were deemed by the model as decision-critical regions for classifying the input time series.\n#### 3. Newly Provided Visual Information\nNow, I have input five additional images to you. Each of these five images is a line plot obtained by converting 1D time series data (corresponding to the five different classes respectively) into 2D visualization. Each plot follows the same axis definition: x-axis for time steps and y-axis for time series values, and there are no red-marked regions in these five images.\n#### 4. Tasks to Perform\nFirst, by comparing the line plot (with red-marked regions) provided in the previous conversation with these five newly provided line plots (each corresponding to one class), analyze in detail why the black-box model classified the time series in the previous conversation’s image into Class 2 (Butterfly) or Class 3 (Fish). Your analysis should focus on comparing the time series features (e.g., value fluctuation patterns, trend change points, time-step distribution of key values) of the previous image (especially the red-marked regions) with those of the five newly provided class-specific line plots, and explain how these comparative features led the model to make the classification. Second, based on this cross-comparison of the six line plots (five class-specific plots plus the previous plot), assess whether there is a possibility that the black-box model made a classification error. If such an error is possible, make the final classification ,explain the potential reasons and support your reasoning with specific comparative observations from the six line plots."
    conversation = round_2(conversation, second_prompt, image_data_list2)
    print("第二轮回答：")
    print(conversation[-1]["content"])
    print("-" * 80)



