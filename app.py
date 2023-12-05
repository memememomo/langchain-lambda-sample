import os
import time
import json
import re
import logging
from typing import Any
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.chat_models import BedrockChat
from slack_bolt import App
from dotenv import load_dotenv
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, HumanMessage, SystemMessage

region="us-east-1"

# ログ
SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# チャット更新間隔
CHAT_UPDATE_INTERVAL_SEC=1

# .envファイルを読み込み
load_dotenv()

# Slack Appインスタンスを生成
app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ.get("SLACK_BOT_TOKEN"),
    process_before_response=True
)

# Slackストリーミングハンドラ
class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        self.update_count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.message += token

        now = time.time()
        if now - self.last_send_time > self.interval:
            app.client.chat_update(
                channel=self.channel,
                ts=self.ts,
                text=f"{self.message}\n\nTyping..."
            )
            self.last_send_time = now
            self.update_count += 1

            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        message_context = "Claude2で生成される情報は不正確または不適切な場合がありますが、当社の見解を述べるものではありません。"

        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
            {"type": "divider"},
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": message_context}],
            },
        ]

        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
            blocks=message_blocks
        )

def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub("<@.*>", "", event["text"])

    id_ts = event["ts"]
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"]

    history = DynamoDBChatMessageHistory(
        table_name=os.environ["DYNAMO_TABLE"],
        session_id=id_ts
    )

    system_message = "You are a good assistant."

    messages = [SystemMessage(content=system_message)]
    messages.extend(history.messages)
    messages.append(HumanMessage(content=message))
    history.add_user_message(message)

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)

    llm = BedrockChat(
        model_id="anthropic.claude-v2",
        streaming=True,
        callbacks=[callback],
        region_name="us-east-1",
        model_kwargs={
            "max_tokens_to_sample": 4000
        }
    )

    ai_message = llm(messages)
    history.add_message(ai_message)

def just_ack(ack):
    ack()

app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

def handler(event, context):
   logging.info("handler clear")
   header = event["headers"]
   logging.info(json.dumps(header))

   if "x-slack-retry-num" in header:
       logging.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
       return 200

   slack_handler = SlackRequestHandler(app=app)
   return slack_handler.handle(event, context)
