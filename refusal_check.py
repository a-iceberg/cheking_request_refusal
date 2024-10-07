import logging
import os
import json

import psycopg

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Literal

from file_service import FileService
from config_manager import ConfigManager


class Step(BaseModel):
    explanation: str
    output: str

class RefusalReasons(BaseModel):
    reason: Literal[self.config_manager.get("reasons").values()]

class ReasonResponse(BaseModel):
    reasons: RefusalReasons
    steps: list[Step]
    suitable_reason: str

class Application:
    def __init__(self):
        self.logger = self.setup_logging()
        self.auth_manager = ConfigManager(
            "./data/auth.json",
            self.logger
        )
        self.config_manager = ConfigManager(
            "./data/config.json",
            self.logger
        )
        self.set_keys()
        self.app = FastAPI()
        self.conn = psycopg.connect(
			dbname="voice_ai",
			user=os.environ.get("DB_USER", ""),
			password=os.environ.get("DB_PASSWORD", ""),
			host=os.environ.get("DB_HOST", ""),
			port=os.environ.get("DB_PORT", ""),
            autocommit=True
		)
        self.setup_routes()
        self.SEED = 654321
        self.CHANNEL_ID = os.environ.get("CHANNEL_ID", "")

    def set_keys(self):
        os.environ["1С_TOKEN"] = self.auth_manager.get("1С_TOKEN", "")
        os.environ["1C_LOGIN"] = self.auth_manager.get("1C_LOGIN", "")
        os.environ["1C_PASSWORD"] = self.auth_manager.get("1C_PASSWORD", "")
        os.environ["DB_USER"] = self.auth_manager.get("DB_USER", "")
        os.environ["DB_PASSWORD"] = self.auth_manager.get("DB_PASSWORD", "")
        os.environ["DB_HOST"] = self.auth_manager.get("DB_HOST", "")
        os.environ["DB_PORT"] = self.auth_manager.get("DB_PORT", "")
        os.environ["OPENAI_API_KEY"] = self.auth_manager.get(
            "OPENAI_API_KEY",
            ""
        )
        os.environ["ANTHROPIC_API_KEY"] = self.auth_manager.get(
            "ANTHROPIC_API_KEY",
            ""
        )
        os.environ["REASON_TOKEN"] = self.auth_manager.get(
            "REASON_TOKEN",
            ""
        )
        self.logger.info("Auth data set successfully")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def text_response(self, text):
        return JSONResponse(content={"type": "text", "body": str(text)})

    def setup_routes(self):
        # Endpoint for checking reguest refusal
        @self.app.get("/refusal_check/{received_token}/{bid_id}/{master_reason_id}")
        async def reasons_comparsion(
            received_token: str,
            bid_id: str,
            master_reason_id: str
        ):
            correct_token = os.environ.get("REASON_TOKEN", "")
            if received_token != correct_token:
                answer = "Неверный токен запроса причины отказа"
                return self.text_response(answer)
            
            query = """
            WITH linked_calls AS (
                SELECT DISTINCT
                    linkedid,
                    call_date
                FROM calls
                WHERE bid_id = %s
            ),
            sorted_transcriptions AS (
                SELECT 
                    t.linkedid,
                    t.start,
                    t.text,
                    lc.call_date
                FROM transcribations t
                JOIN linked_calls lc ON t.linkedid = lc.linkedid
                WHERE t.text IS NOT NULL AND t.text <> ''
                ORDER BY lc.call_date, t.linkedid, t.start
            )
            SELECT linkedid, text
            FROM sorted_transcriptions;
            """

            try:
                with self.conn.cursor() as cur:
                    cur.execute(query, (bid_id,))
                    results = cur.fetchall()

                    conversations = {}
                    for linkedid, text in results:
                        if linkedid not in conversations:
                            conversations[linkedid] = []
                        conversations[linkedid].append(text)

                    full_text = []
                    for linkedid, texts in conversations.items():
                        conversation_text = ". ".join(
                            text.strip() for text in texts
                        )
                        full_text.append(
                            f"Следующий диалог в разговоре: {conversation_text}"
                        )
                    final_text = "\n".join(full_text)
            except Esxeption as e:
                self.logger.error(f"Ошибка при работе с базой данных: {e}")

            try:
                if self.company == "OpenAI":
                    client = AsyncOpenAI(
                        api_key=os.environ.get("OPENAI_API_KEY", "")
                    )
                    temperature = self.config_manager.get("openai_temperature")
                    messages = [
                        {
                            "role": "system",
                            "content": """Вы специалист маркетингу и работе с возражениями, отказами клиентов, а также хороший социолог.
Далее представлены диалоги в рамках одной заявки в сервисный центр. Проанализируйте их все и выберите наиболее подходящую причину отказа по данной заявке из имеющегося у вас списка, если в диалогах действительно точно фигурировало именно то, что вы выбираете, например, если выбрали "дорого", то точно должна быть информация именно про дорого в разговоре, а не просто местоположение или что-то ещё.
Если в диалогах никак не фигурировал отказ или диалоги в принципе отсутствуют, выбирайте соответствующую причину из представленных - "Не было отказа".
Возвращайте только именно текст самой причины дословно, именно так, как она была записана изначально в списке.""",
                        },
                        {"role": "user", "content": final_text}
                    ]
                    response = await client.beta.chat.completions.parse(
                        model=self.config_manager.get("openai_model"),
                        temperature=temperature,
                        seed=self.SEED,
                        response_format=ReasonResponse,
                        messages=messages
                    )
                    llm_reason = response.choices[0].message
                    if llm_reason.parsed:
                        llm_reason = llm_reason.parsed.suitable_reason
                    else:
                        messages = [
                        {
                            "role": "system",
                            "content": """Вы специалист маркетингу и работе с возражениями, отказами клиентов, а также хороший социолог.
Далее представлены диалоги в рамках одной заявки в сервисный центр. Проанализируйте их все и выберите наиболее подходящую причину отказа по данной заявке из списка ниже, если в диалогах действительно точно фигурировало именно то, что вы выбираете, например, если выбрали "дорого", то точно должна быть информация именно про дорого в разговоре, а не просто местоположение или что-то ещё.
Если в диалогах никак не фигурировал отказ или диалоги в принципе отсутствуют, выбирайте соответствующую причину из представленных - "Не было отказа".
Возвращайте только именно текст самой причины дословно, именно так, как она была записана изначально в списке.
Список возможных причин:
Выбранную услугу, ремонт уже оказал или оказывает другой человек, мастер, знакомый и тому подобное
Клиент посчитал, что для него дорого стоит именно сама диагностика, её цена большая, не что-то ещё другое; должна обязательно фигурировать сама оцененная стоимость диагностики и быть зафиксировано это до прибытия мастера на адрес
Техника заработала, неполадка исправилась, проблема решена, не требует вмешательства
Клиент посчитал, что дорого, большая цена за сами работы, ремонт, услуги, в том числе с возможно предложенной скидкой; должна обязательно фигурировать сама оцененная стоимость работ и быть зафиксировано это до прибытия мастера на адрес
Оказался сложный ремонт, крупногабаритная техника или ещё какая-то причина, по которой ремонт осуществляется только в стационарном сервисном центре, не на дому
Клиент посчитал, что большая цена за оплату именно выезда мастера по его дальнему адресу, дорогой выезд, не сами услуги / ремонт; должна обязательно фигурировать сама оцененная стоимость выезда и быть зафиксировано это до прибытия мастера на адрес
Был осуществлен дублирующий вызов по одной и той же услуге, технике по тому же адресу в то же время; именно повторный вызов после работ в прошлый раз не подходит, сюда не относится
Заявка по безналичному расчету, оформляется в другом отделе
Был неправильно указан телефон и / или адрес заказчика
Гарантийный ремонт по гарантии именно производителя, не сервисного центра
Не успели привезти необходимую технику (подходит, только если требовалась установка техники или если речь о кондиционерах)
Вместо ремонта, услуги клиент решил купить новую технику на замену и обязательно уже оплатил её, ждёт доставку или уже получил. Ещё только намерение купить сюда не относится
Компания не осуществляет запрошенные услуги / ремонт конкретной нужной техники
Техника была разобрана и находится в этом же состоянии или собранном частично; Заявка на сборку
Не подготовлено должным образом место для работы: недостаточная площадь помещения, техника заставлена предметами, встроена в дорогую мебель и тому подобное
Антисанитарные условия на адресе: насекомые, сильная грязь, захламлённость и тому подобное, помещение требует предварительной санитарной обработки
Именно клиент и мастер не смогли договориться именно друг с другом сами об удобном времени визита
Мастер по той или иной причине не поехал к клиенту, хотя должен был в уже обязательно оговоренное время. Также мог изначально  в том числе даже не позвонить клиенту
Клиент бросает трубку, не хочет разговаривать. Только после того, как поднимет трубку изначально и что-то услышит
Мастер не может дозвониться до клиента, не берут трубку. Либо же клиент сам уточнил, что не сможет дождаться мастера, не сможет быть по указанному адресу и тому подобное
Нет нужных деталей на складе, либо слишком долго ждать их поставки, либо нет в принципе возможности заказать нужные детали
Недоступны нужные детали по причине именно снятия их с производства производителем, так как техника устарела
Запрос по неизвестной технике, по которой нет документации
Не было отказа""",
                        },
                        {"role": "user", "content": final_text}
                        ]
                        response = await client.chat.completions.create(
                            model=self.config_manager.get("reserve_openai_model"),
                            temperature=temperature,
                            seed=self.SEED,
                            messages=messages
                        )
                        llm_reason = response.choices[0].message.content

                elif self.company == "Anthropic":
                    client = AsyncAnthropic(
                        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                    )
                    response = await client.messages.create(
                        model=self.config_manager.get("anthropic_model"),
                        temperature=self.config_manager.get("anthropic_temperature"),
                        system="""Вы специалист маркетингу и работе с возражениями, отказами клиентов, а также хороший социолог.
Далее представлены диалоги в рамках одной заявки в сервисный центр. Проанализируйте их все и выберите наиболее подходящую причину отказа по данной заявке из списка ниже, если в диалогах действительно точно фигурировало именно то, что вы выбираете, например, если выбрали "дорого", то точно должна быть информация именно про дорого в разговоре, а не просто местоположение или что-то ещё.
Если в диалогах никак не фигурировал отказ или диалоги в принципе отсутствуют, выбирайте соответствующую причину из представленных - "Не было отказа".
Возвращайте только именно текст самой причины дословно, именно так, как она была записана изначально в списке.
Список возможных причин:
Выбранную услугу, ремонт уже оказал или оказывает другой человек, мастер, знакомый и тому подобное
Клиент посчитал, что для него дорого стоит именно сама диагностика, её цена большая, не что-то ещё другое; должна обязательно фигурировать сама оцененная стоимость диагностики и быть зафиксировано это до прибытия мастера на адрес
Техника заработала, неполадка исправилась, проблема решена, не требует вмешательства
Клиент посчитал, что дорого, большая цена за сами работы, ремонт, услуги, в том числе с возможно предложенной скидкой; должна обязательно фигурировать сама оцененная стоимость работ и быть зафиксировано это до прибытия мастера на адрес
Оказался сложный ремонт, крупногабаритная техника или ещё какая-то причина, по которой ремонт осуществляется только в стационарном сервисном центре, не на дому
Клиент посчитал, что большая цена за оплату именно выезда мастера по его дальнему адресу, дорогой выезд, не сами услуги / ремонт; должна обязательно фигурировать сама оцененная стоимость выезда и быть зафиксировано это до прибытия мастера на адрес
Был осуществлен дублирующий вызов по одной и той же услуге, технике по тому же адресу в то же время; именно повторный вызов после работ в прошлый раз не подходит, сюда не относится
Заявка по безналичному расчету, оформляется в другом отделе
Был неправильно указан телефон и / или адрес заказчика
Гарантийный ремонт по гарантии именно производителя, не сервисного центра
Не успели привезти необходимую технику (подходит, только если требовалась установка техники или если речь о кондиционерах)
Вместо ремонта, услуги клиент решил купить новую технику на замену и обязательно уже оплатил её, ждёт доставку или уже получил. Ещё только намерение купить сюда не относится
Компания не осуществляет запрошенные услуги / ремонт конкретной нужной техники
Техника была разобрана и находится в этом же состоянии или собранном частично; Заявка на сборку
Не подготовлено должным образом место для работы: недостаточная площадь помещения, техника заставлена предметами, встроена в дорогую мебель и тому подобное
Антисанитарные условия на адресе: насекомые, сильная грязь, захламлённость и тому подобное, помещение требует предварительной санитарной обработки
Именно клиент и мастер не смогли договориться именно друг с другом сами об удобном времени визита
Мастер по той или иной причине не поехал к клиенту, хотя должен был в уже обязательно оговоренное время. Также мог изначально  в том числе даже не позвонить клиенту
Клиент бросает трубку, не хочет разговаривать. Только после того, как поднимет трубку изначально и что-то услышит
Мастер не может дозвониться до клиента, не берут трубку. Либо же клиент сам уточнил, что не сможет дождаться мастера, не сможет быть по указанному адресу и тому подобное
Нет нужных деталей на складе, либо слишком долго ждать их поставки, либо нет в принципе возможности заказать нужные детали
Недоступны нужные детали по причине именно снятия их с производства производителем, так как техника устарела
Запрос по неизвестной технике, по которой нет документации
Не было отказа""",
                        messages=[
                            {"role": "user", "content": final_text},
                        ]
                    )
                    llm_reason = response.content.text
            except Exception as e:
                self.logger.error(f"Error in getting refusal reason: {e}")
            
            master_reason = self.config_manager.get("reasons")[master_reason_id]
            answer = {"master_reason": master_reason, "llm_reason": llm_reason}

            return JSONResponse(
                content=json.dumps(answer, ensure_ascii=False)
            )


application = Application()
app = application.app