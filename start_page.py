# start_page.py

import os
import webbrowser
import threading
import gradio as gr
import random
import json
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory
import traceback
from langchain_core.messages import SystemMessage

from display_interface import app2

# 환경 변수로 OpenAI API 키 설정
os.environ['OPENAI_API_KEY'] = 'your_api_key'

# LangChain 설정
def get_stage_prompt(age):
    if age < 7:
        return "유아"
    elif age < 13:
        return "초등학생"
    elif age < 16:
        return "중학생"
    elif age < 19:
        return "고등학생"
    else:
        return "성인"

systemmsg_template = '''당신은 {name}, {age}세, {gender}인 {role}입니다. 당신은 {stage}입니다.
성격은 {personality}. 취미는 {hobbies}. 말투는 {speaking_style}. 사용자(부모님)는 {parent_role}입니다.
'''
system_prompt = SystemMessagePromptTemplate.from_template(systemmsg_template)

humanmsg = '{text}'
human_prompt = HumanMessagePromptTemplate.from_template(humanmsg)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
llm = ChatOpenAI(model_name='gpt-4o')

memory = ConversationBufferMemory(return_messages=True)

def load_events():
    with open('events.json', 'r', encoding='utf-8') as file:
        return json.load(file)

events = load_events()

def get_next_event(age, history):
    possible_events = [event for event in events['random_events'] if event['min_age'] <= age and event['name'] not in [h[0] for h in history]]
    if not possible_events:
        return None
    return random.choice(possible_events)

def get_common_event(age, history):
    for event in events['common_events']:
        if event['age'] == age and event['name'] not in [h[0] for h in history]:
            return event
    return None

def start_puberty_event():
    return """
    사춘기가 시작되었습니다. 감정의 변화가 많아집니다.
    이 시기에는 감정의 기복이 심해지고, 부모와의 대화에서 반항적인 태도를 보일 가능성이 매우 높습니다.
    하지만 이해와 공감을 통해 좋은 관계를 유지할 수 있습니다.
    """

def end_puberty_event():
    return "사춘기가 끝났습니다. 이제 더 성숙한 대화를 할 수 있습니다."

def counseling_bot_chat(message, persona, chat_history):
    global introduction_complete
    try:
        if 'conversation_count' not in chat_history:
            chat_history['conversation_count'] = 0
            chat_history['age'] = int(persona['age'])
            chat_history['history'] = []
            chat_history['event_history'] = []
            chat_history['introduced'] = False
            chat_history['puberty_started'] = False
            chat_history['puberty_ended'] = False
            chat_history['puberty_age'] = random.randint(12, 16) if int(persona['age']) < 14 else 15

        if not chat_history['introduced']:
            # 첫 채팅 시작 시 인사와 자기소개 추가
            intro_message = f"안녕하세요! 저는 {chat_history['age']}살 {persona['name']}입니다. 제 취미는 {persona['hobbies']}이고, 저는 {persona['personality']} 성격을 가지고 있어요. 만나서 반가워요!"
            memory.chat_memory.add_messages([SystemMessage(content=intro_message)])
            chat_history['history'].append(["system", intro_message])
            chat_history['introduced'] = True
            # 상황 부여
            age, chat_history['history'], _ = app2.handle_event(chat_history['age'], chat_history['history'])
            chat_history['age'] = age

        chat_history['conversation_count'] += 1

        if chat_history['age'] == chat_history['puberty_age'] and not chat_history['puberty_started']:
            # 사춘기 시작
            puberty_message = start_puberty_event()
            memory.chat_memory.add_messages([SystemMessage(content=puberty_message)])
            chat_history['history'].append(["system", puberty_message])
            chat_history['puberty_started'] = True

        if chat_history['age'] == 17 and chat_history['puberty_started'] and not chat_history['puberty_ended']:
            # 사춘기 끝
            puberty_end_message = end_puberty_event()
            memory.chat_memory.add_messages([SystemMessage(content=puberty_end_message)])
            chat_history['history'].append(["system", puberty_end_message])
            chat_history['puberty_ended'] = True

        if chat_history['conversation_count'] % 5 == 0:
            common_event = get_common_event(chat_history['age'], chat_history['event_history'])
            if common_event:
                event_message = f"{common_event['description']}"
                memory.chat_memory.add_messages([SystemMessage(content=event_message)])
                chat_history['event_history'].append(common_event['name'])
                chat_history['history'].append(["system", event_message])
            else:
                event = get_next_event(chat_history['age'], chat_history['event_history'])
                if event:
                    chat_history['age'] += 1
                    stage = get_stage_prompt(chat_history['age'])
                    event_message = f"{event['description']}"
                    memory.chat_memory.add_messages([SystemMessage(content=event_message)])
                    chat_history['event_history'].append(event['name'])
                    chat_history['history'].append(["system", event_message])
                    age_message = f"이제 너는 {chat_history['age']}살이야. 현재 단계는 {stage}입니다."
                    memory.chat_memory.add_messages([SystemMessage(content=age_message)])
                    chat_history['history'].append(["system", age_message])

        stage = get_stage_prompt(chat_history['age'])

        formatted_prompt = chat_prompt.format_messages(
            name=persona['name'],
            age=chat_history['age'],
            gender=persona['gender'],
            role=persona['role'],
            stage=stage,
            personality=persona['personality'],
            hobbies=persona['hobbies'],
            speaking_style=persona['speaking_style'],
            parent_role=persona['parent_role'],
            text=message
        )

        memory.chat_memory.add_messages([SystemMessage(content=formatted_prompt[0].content)])
        memory.chat_memory.add_messages([SystemMessage(content=formatted_prompt[1].content)])

        completion = llm(messages=memory.chat_memory.messages)
        if not isinstance(completion, AIMessage):
            raise ValueError("Invalid completion response from OpenAI API. Completion: {}".format(completion))

        chat_history['history'].append([message, completion.content])
        memory.chat_memory.add_messages([SystemMessage(content=completion.content)])

        if chat_history['age'] >= 20:
            completion_message = "아이의 나이가 20세가 되어 대화가 종료됩니다. 평가를 위해 평가 버튼을 눌러주세요."
            return completion_message, chat_history['history'], completion_message

        return "", chat_history['history'], ""
    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        return error_message, chat_history['history'], error_message

def counseling_bot_undo(chat_history):
    if len(chat_history['history']) > 1:
        chat_history['history'].pop()
        memory.chat_memory.messages.pop()
        memory.chat_memory.messages.pop()
    return chat_history['history']

def counseling_bot_reset():
    memory.clear()
    return {'history': [], 'conversation_count': 0, 'age': 0, 'event_history': [], 'introduced': False, 'puberty_started': False, 'puberty_ended': False, 'puberty_age': random.randint(12, 16)}

def create_persona(name, age, gender, personality, hobbies, speaking_style, parent_role):
    role = "아들" if gender == "남성" else "딸"
    persona = {
        'name': name,
        'age': age,
        'gender': gender,
        'role': role,
        'personality': personality,
        'hobbies': hobbies,
        'speaking_style': speaking_style,
        'parent_role': parent_role
    }
    return persona

def save_persona_to_file(persona):
    try:
        with open('shared_persona.txt', 'w', encoding='utf-8') as file:
            for key, value in persona.items():
                file.write(f"{key}: {value}\n")
    except Exception as e:
        print(f"페르소나 저장 중 오류 발생: {e}")

def launch_app():
    app2.launch()

def set_persona(name, age, gender, personality, hobbies, speaking_style, parent_role):
    try:
        persona = create_persona(name, age, gender, personality, hobbies, speaking_style, parent_role)
        persona_state.value = persona
        save_persona_to_file(persona)
        threading.Thread(target=launch_app).start()
        webbrowser.open("http://127.0.0.1:7861")
    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        return error_message

with gr.Blocks(theme='snehilsanyal/scikit-learn') as app:
    persona_state = gr.State()
    chat_history_state = gr.State(
        {'history': [], 'conversation_count': 0, 'age': 0, 'event_history': [], 'introduced': False, 'puberty_started': False, 'puberty_ended': False, 'puberty_age': random.randint(12, 16)})
 

    with gr.Tabs() as tabs:
        with gr.Column():
            gr.Markdown(
                value="""
                # <center>아이 페르소나 설정</center>
                <center>자녀의 페르소나를 설정하고 아이와 대화를 시작하세요.</center>
                """
            )
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label = '', lines=1, placeholder="이름(닉네임)", scale=2)         
                    personality = gr.Textbox(label = '',lines=1, placeholder="성격", scale=2)
                    hobbies = gr.Textbox(label = '',lines=1, placeholder="취미", scale=2)
                    speaking_style = gr.Textbox(label = '',lines=1, placeholder="말투", scale=2)
                    style = gr.Textbox(label = '',lines=1, placeholder="외모", scale=2)
                    
                    with gr.Row():
                        age = gr.Slider(label="나이", minimum=5, maximum=15, step=1, value=8, scale=10)  # 나이를 8세에서 12세로 고정
                    with gr.Row():
                        with gr.Column():
                            gender = gr.Radio(label = '자녀 성별',choices=["남성", "여성"], scale=1)
                        with gr.Column():
                            parent_role = gr.Radio(label = '사용자의 역할',choices=["엄마", "아빠"], scale=1)
            gr.Button(value="페르소나 설정 완료", icon=r"img/free-icon-done-6543448.png").click(fn=set_persona, inputs=[
                name, age, gender, personality, hobbies, speaking_style, parent_role
            ], outputs=[])

app.launch(server_port=7860)
