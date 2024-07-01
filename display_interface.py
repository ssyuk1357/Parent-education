from abc import ABC, abstractmethod
import os
import json
import random
import traceback
import gradio as gr
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# 버전 충돌
# import Chroma_consultant
# from Chroma_consultant import get_empathy_context

# API 키 설정
os.environ['OPENAI_API_KEY'] = 'your_api_key'

# 언어 모델 인스턴스 생성
llm = ChatOpenAI(
    model_name="gpt-4o"
)

# 메모리 인스턴스 생성
memory = ConversationBufferMemory(input_key="input", output_key="output")

# 채팅봇 설정
conversation_count = 0
event_conversation_count = 0
current_event = None
introduction_complete = False
current_system_text = "당신의 자녀가 마주하게 될 이벤트가 준비중입니다."
puberty_event_occurred = False


class Chatbot(ABC):
    def __init__(self, model_name, persona_file):
        self.llm = ChatOpenAI(model_name=model_name)
        self.persona = self.load_persona_from_file(persona_file)
        self.persona_file = persona_file

    def load_persona_from_file(self, persona_file):
        persona = {}
        try:
            with open(persona_file, 'r', encoding='utf-8') as file:
                for line in file:
                    key, value = line.strip().split(': ', 1)
                    if key == 'age' or key == 'conversation_count':
                        persona[key] = int(value)
                    else:
                        persona[key] = value
        except FileNotFoundError:
            print(f"{persona_file} 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"페르소나 로드 중 오류 발생: {e}")
        return persona

    def save_persona_to_file(self):
        try:
            with open(self.persona_file, 'w', encoding='utf-8') as file:
                for key, value in self.persona.items():
                    file.write(f"{key}: {value}\n")
        except Exception as e:
            print(f"페르소나 저장 중 오류 발생: {e}")

    @abstractmethod
    def chat(self, user_input, history):
        pass

    @abstractmethod
    def reset(self):
        pass


class PromptChatbot(Chatbot):
    def __init__(self, model_name, persona_file):
        super().__init__(model_name, persona_file)

    def get_combined_prompt(self, user_input):
        systemmsg = f'''
        당신은 상담사 육은영입니다. 당신의 성격은 {self.persona['personality']}입니다.
        당신의 역할은 {self.persona['role']}로서 사용자의 메시지를 평가하는 것입니다.
        '''

        purpose_prompt = f'''
        사용자로부터 받은 메시지를 평가하고, 해당 메시지에 대한 적절한 피드백을 제공합니다.
        '''

        human_prompt = f'''
        사용자 메시지: "{user_input}"
        '''
        return systemmsg + purpose_prompt + human_prompt

    def chat(self, user_input, history):
        try:
            combined_prompt = self.get_combined_prompt(user_input)
            system_prompt = SystemMessagePromptTemplate.from_template(combined_prompt)
            humanmsg = '{text}'
            human_prompt = HumanMessagePromptTemplate.from_template(humanmsg)

            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            prompt = chat_prompt.format_messages(
                name=self.persona['name'],
                age=self.persona['age'],
                gender=self.persona['gender'],
                role=self.persona['role'],
                personality=self.persona['personality'],
                hobbies=self.persona['hobbies'],
                speaking_style=self.persona['speaking_style'],
                text=user_input
            )

            response = self.llm(messages=prompt).content

            history.append((user_input, response))

            return response, history
        except Exception as e:
            error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
            return error_message, history

    def reset(self):
        persona = self.persona
        gender = persona.get('gender', '여성')
        age = int(persona.get('age', 0))

        if gender == '남성':
            if age >= 10:
                image_path = r'img/boy_10_above.webp'
            else:
                image_path = r'img/boy_10_below.webp'
        else:
            if age >= 10:
                image_path = r'img/girl_10_above.webp'
            else:
                image_path = r'img/girl_10_below.webp'

        return [], image_path


def load_persona_from_file():
    persona = {}
    try:
        with open('shared_persona.txt', 'r', encoding='utf-8') as file:
            for line in file:
                key, value = line.strip().split(': ', 1)
                persona[key] = value
    except FileNotFoundError:
        print("shared_persona.txt 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"페르소나 로드 중 오류 발생: {e}")
    return persona


def save_persona_to_file(persona):
    try:
        with open('shared_persona.txt', 'w', encoding='utf-8') as file:
            for key, value in persona.items():
                file.write(f"{key}: {value}\n")
    except Exception as e:
        print(f"페르소나 저장 중 오류 발생: {e}")


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


def get_combined_prompt(persona, history):
    stage = get_stage_prompt(int(float(persona['age'])))  # float 변환 후 int로 변환
    systemmsg = f'''
    당신은 {persona['name']}, {persona['age']}세, {persona['gender']}인 {persona['role']}입니다.
    당신은 {stage}입니다. 성격은 {persona['personality']}. 취미는 {persona['hobbies']}.
    말투는 {persona['speaking_style']}. 사용자(부모님)는 {persona['parent_role']}입니다.
    '''

    if stage == "유아":
        response_style = "간단한 단어와 짧은 문장으로 반응하세요."
    elif stage == "초등학생":
        response_style = "좀 더 복잡한 문장과 논의가 가능하게 반응하세요."
    elif stage == "중학생":
        response_style = "감정 표현과 논의를 포함한 대화를 하세요."
    elif stage == "고등학생":
        response_style = "성숙한 어휘와 복잡한 감정을 포함한 대화를 하세요."
    else:
        response_style = "성인답게 성숙하고 논리적인 대화를 하세요."

    purpose_prompt = f'''
    당신은 {stage}입니다. {response_style}
    당신이 보낸 메시지의 감정을 분석해서 그 메시지에 맞는 이모지를 넣어 주세요.
    감정 이모지의 종류는 "😊 (기쁨), 😳 (당황), 😠 (분노), 😰 (불안), 😢 (상처), 😔 (슬픔)"이 있고 이중에서 해당하는 이모지 하나를 문단의 마지막에 넣어주세요.
    감정 이모지가 아닌 다른 이모지는 상황과 메세지에 맞게 자유롭게 넣어주세요.
    당신의 나이가 증가함에 따라 증가된 나이에 맞는 어휘를 구사 하세요.
    '''

    situation_prompt = '''
    이벤트가 발생되고 당신이 이벤트 상황을 인지한다면, 이벤트 상황에 몰입해서 사용자(부모)와 소통을 이어가주세요.
    사춘기 이벤트의 경우, 당신은 평소보다 감정의 기복이 심해지고 반항적인 태도를 보일 가능성이 있습니다. 
    하지만, 여전히 당신의 본래 성격이 남아있으므로 상황에 따라 다정함과 반항적인 태도를 번갈아 보일 수 있습니다.
    이벤트는 5번의 대화동안 유지되고, 새로운 이벤트로 최신화 됩니다. 상황이 변함에 따라 맞춰서 소통해주세요.
    '''

    history_prompt = f'''
    당신의 목표는 대화 기록({history})을 바탕으로 자연스럽게 대화를 이어 가는 것 입니다.
    또한, 사용자(부모)와 대화를 하면서 발생되는 이벤트에 부모와 함께 소통하며 반응하고 성인이 되었을 때 대화를 종료합니다.
    그 후, 여태까지 사용자(부모)와 나누었던 모든 대화를 바탕으로 사용자(부모)가 어떤 부모였는지 편지를 작성해서 보여 주세요.
    편지에는 부모에게 감사했던 점이나 부족했던 점을 포함하여, 자녀의 입장에서 솔직하게 느낀 감정과 생각을 담아주세요.
    이 편지는 부모가 자녀와의 대화를 돌아보고, 앞으로의 관계를 더욱 좋게 발전시킬 수 있도록 돕는 역할을 합니다.
    '''
    return systemmsg + purpose_prompt + situation_prompt + history_prompt


def handle_event(age, history):
    global conversation_count, event_conversation_count, current_event, current_system_text, puberty_event_occurred
    if 11 <= age <= 14 and not puberty_event_occurred:
        event = {
            "name": "사춘기",
            "description": "사춘기가 시작되었습니다. 감정의 변화가 많아집니다. 이 시기에는 감정의 기복이 심해지고, 부모와의 대화에서 반항적인 태도를 보일 가능성이 매우 높습니다. 하지만 이해와 공감을 통해 좋은 관계를 유지할 수 있습니다."
        }
        puberty_event_occurred = True
    else:
        event = get_common_event(age, history) or get_next_event(age, history)

    if event:
        current_event = event
        stage = get_stage_prompt(age)
        event_message = f"챕터: {event['name']}\n설명: {event['description']}"
        current_system_text = event_message
        memory.save_context({"input": "system"}, {"output": event_message})
        event_conversation_count = 0
    return age, history, current_system_text


def get_common_event(age, history):
    with open('events.json', 'r', encoding='utf-8') as file:
        events = json.load(file)
    for event in events['common_events']:
        if event['age'] == age and event['name'] not in [h[0] for h in history]:
            return event
    return None


def get_next_event(age, history):
    with open('events.json', 'r', encoding='utf-8') as file:
        events = json.load(file)
    possible_events = [event for event in events['random_events'] if
                       event['min_age'] <= age and event['name'] not in [h[0] for h in history]]
    if not possible_events:
        return None
    return random.choice(possible_events)


def increment_age_and_handle_event(persona, history):
    age = float(persona['age'])
    age += 1  # 나이를 증가시킴
    age, history, current_system_text = handle_event(age, history)
    persona['age'] = str(age)
    save_persona_to_file(persona)
    return history, current_system_text, age


def chat_langchain(user_input, history):
    global conversation_count, event_conversation_count, current_event, introduction_complete, current_system_text
    try:
        persona = load_persona_from_file()
        age = float(persona['age'])

        gender = persona.get('gender', '여성')
        if gender == '남성':
            if age >= 10:
                image_path = r'img/boy_10_above.webp'
            else:
                image_path = r'img/boy_10_below.webp'
        else:
            if age >= 10:
                image_path = r'img/girl_10_above.webp'
            else:
                image_path = r'img/girl_10_below.webp'

        if not introduction_complete:
            introduction_complete = True
            introduction_message = f"안녕하세요! 저는 {age}살 {persona['name']}입니다. 제 취미는 {persona['hobbies']}이고, 저는 {persona['personality']} 성격을 가지고 있어요. 만나서 반가워요😊"
            memory.save_context({"input": user_input}, {"output": introduction_message})
            history.append((user_input, introduction_message))
            conversation_count += 1

            return "", history, image_path, current_system_text, age

        conversation_count += 1
        event_conversation_count += 1

        combined_prompt = get_combined_prompt(persona, history)
        system_prompt = SystemMessagePromptTemplate.from_template(combined_prompt)
        humanmsg = '{text}'
        human_prompt = HumanMessagePromptTemplate.from_template(humanmsg)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        prompt = chat_prompt.format_messages(
            name=persona['name'],
            age=age,
            gender=persona['gender'],
            role=persona['role'],
            personality=persona['personality'],
            hobbies=persona['hobbies'],
            speaking_style=persona['speaking_style'],
            parent_role=persona['parent_role'],
            text=user_input
        )

        response = llm(messages=prompt).content

        memory.save_context({"input": user_input}, {"output": response})

        history.append((user_input, response))

        if conversation_count == 1 and current_event is None:
            age, history, current_system_text = handle_event(age, history)
            return "", history, image_path, current_system_text, age

        if event_conversation_count >= 5:
            if current_event:
                details = current_event.get('details', '')
                questions = current_event.get('questions', [])
                details_message = f"상세 설명: {details}\n예시 질문: {', '.join(questions)}"
                memory.save_context({"input": "system"}, {"output": details_message})
                current_event = None

        if conversation_count % 5 == 0 and not current_event:
            history, current_system_text, age = increment_age_and_handle_event(persona, history)

        if float(persona['age']) >= 20:
            completion_message = "아이의 나이가 20세가 되어 대화가 종료됩니다. 평가를 위해 평가 버튼을 눌러주십시오."
            history.append(("system", completion_message))
            return completion_message, history, image_path, "", 20

        return "", history, image_path, current_system_text, age
    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        return error_message, history, error_message, "", age


def counseling_bot_reset():
    global conversation_count, event_conversation_count, current_event, introduction_complete, current_system_text, puberty_event_occurred
    conversation_count = 0
    event_conversation_count = 0
    current_event = None
    introduction_complete = False
    current_system_text = "당신의 자녀가 마주하게 될 이벤트가 준비중입니다."
    memory.clear()
    persona = load_persona_from_file()
    initial_age = float(persona['age']) if 'age' in persona else 5
    puberty_event_occurred = False

    gender = persona.get('gender', '여성')
    if gender == '남성':
        if initial_age >= 10:
            image_path = r'img/boy_10_above.webp'
        else:
            image_path = r'img/boy_10_below.webp'
    else:
        if initial_age >= 10:
            image_path = r'img/girl_10_above.webp'
        else:
            image_path = r'img/girl_10_below.webp'

    return [], image_path, gr.update(interactive=True), current_system_text, initial_age


chatbot2 = PromptChatbot(model_name='gpt-4o', persona_file='shared_persona2.txt')


def evaluate_response(history1):
    if history1:
        kid_response = history1[-1][1]
        # 버전 충돌로 인한 제외
        # empathy_response = get_empathy_context(kid_response)
        #command_prompt = f"'{kid_response}'는 자녀인 챗봇의 대화이고 '{empathy_response}' 자녀인 챗봇 대화내용을 바탕으로 벡터DB로 찾은 공감형대화셋에서 가장 유사도 높은 대화내역들이야 이걸 바탕으로 아빠가 어떻게 말해야 공감형 대화를 할 수 있는지에 대해 대화방식,대화예시로  총 3줄 요약으로 답해줘"

        #response, _ = chatbot2.chat(command_prompt, [])

        return response
    return "버전 충돌로 인해 구현중입니다."


with gr.Blocks(theme=gr.themes.Base()) as app2:
    with gr.Tab("멋진 아빠 되기 프로젝트"):
        gr.Markdown(
            value="""
            # <center>멋진 아빠 되기 프로젝트</center>
            <center>당신은 한 아이의 아빠입니다. 잘 키워서 아이가 꿈을 이룰 수 있게 해주세요.</center>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    system_text = gr.Textbox(
                        scale=9,
                        label='상황',
                        placeholder='당신의 자녀가 마주하게 될 이벤트가 준비중입니다.',
                        interactive=False
                    )
                    system_age = gr.Textbox(
                        scale=1,
                        label='나이',
                        placeholder='0',
                        interactive=False
                    )

                cb_chatbot = gr.Chatbot(
                    min_width=300,
                    scale=5,
                    label="자녀와의 대화",
                    value=[[None, '안녕하세요. 대화를 하기 전에 먼저 저를 소개 하고 싶어요!']],
                    show_label=False
                )
            with gr.Column(scale=1):
                persona = load_persona_from_file()
                gender = persona.get('gender', '여성')
                age = int(persona.get('age', 0))

                if gender == '남성':
                    if age >= 10:
                        initial_image_path = r'img/boy_10_above.webp'
                    else:
                        initial_image_path = r'img/boy_10_below.webp'
                else:
                    if age >= 10:
                        initial_image_path = r'img/girl_10_above.webp'
                    else:
                        initial_image_path = r'img/girl_10_below.webp'

                sub_image = gr.Image(
                    label="당신의 자녀",
                    value=initial_image_path,
                    interactive=False
                )
                sub_text = gr.Textbox(
                    label="상담사가가 당신을 바라보며 말합니다.",
                    interactive=False
                )

        with gr.Row():
            cb_user_input = gr.Textbox(
                lines=1,
                scale=6,
                placeholder="입력 창",
                container=False
            )
            cb_send_btn = gr.Button(
                value="보내기",
                icon=r"img/free-icon-conversation-1206850.png",
                scale=1,
                variant="primary"
            )
            sub_send_btn = gr.Button(
                value="상담사의 평가",
                scale=1,
                variant="primary"
            ).click(
                fn=evaluate_response,
                inputs=[cb_chatbot],
                outputs=[sub_text]
            )

        with gr.Row():
            gr.Button(value="마지막 대화 저장", icon=r'img/free-icon-txt-file-2267023.png').click(
                fn=counseling_bot_reset,
                inputs=[],
                outputs=[cb_chatbot, sub_image, cb_user_input, system_text, system_age]
            )
            cb_send_btn.click(
                fn=chat_langchain,
                inputs=[cb_user_input, cb_chatbot],
                outputs=[cb_user_input, cb_chatbot, sub_image, system_text, system_age]
            )
            cb_user_input.submit(
                fn=chat_langchain,
                inputs=[cb_user_input, cb_chatbot],
                outputs=[cb_user_input, cb_chatbot, sub_image, system_text, system_age]
            )

if __name__ == "__main__":
    app2.launch()
