import random
import time

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(5, 20)

    @task
    def ask_question(self):
        self.client.get("/")
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "content": random.choice(
                            [
                                "Који сервис треба да користим кад ми се роди беба?",
                                "За шта могу да користим портал еуправа?",
                                "Која је адреса еуправе, како да приступим?",
                                "Како да региструјем налог на еуправи?",
                                "Како да добијем родни лист за новорођенче"
                            ]
                        ),
                        "role": "user",
                    },
                ],
                "context": {
                    "overrides": {
                        "retrieval_mode": "hybrid",
                        "semantic_ranker": True,
                        "semantic_captions": False,
                        "top": 3,
                        "suggest_followup_questions": False,
                    },
                },
            },
        )
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "messages": [
                    {"content": "Да ли еуправу могу да користим са мобилног телефона?", "role": "user"},
                    {
                        "content": "Порталу еУправа је могућ приступ, регистрација и употреба сервиса и са мобилних уређаја [portal_euprava.docx].",
                        "role": "assistant",
                    },
                    {"content": "Да ли могу да платим сервис преко портала еУправа?", "role": "user"},
                ],
                "context": {
                    "overrides": {
                        "retrieval_mode": "hybrid",
                        "semantic_ranker": True,
                        "semantic_captions": False,
                        "top": 3,
                        "suggest_followup_questions": True,
                    },
                },
            },
        )


class ChatVisionUser(HttpUser):
    wait_time = between(5, 20)

    @task
    def ask_question(self):
        self.client.get("/")
        time.sleep(5)
        self.client.post(
            "/chat/stream",
            json={
                "messages": [
                    {
                        "content": "Can you identify any correlation between oil prices and stock market trends?",
                        "role": "user",
                    }
                ],
                "context": {
                    "overrides": {
                        "top": 3,
                        "temperature": 0.3,
                        "minimum_reranker_score": 0,
                        "minimum_search_score": 0,
                        "retrieval_mode": "hybrid",
                        "semantic_ranker": True,
                        "semantic_captions": False,
                        "suggest_followup_questions": False,
                        "use_oid_security_filter": False,
                        "use_groups_security_filter": False,
                        "vector_fields": ["embedding", "imageEmbedding"],
                        "use_gpt4v": True,
                        "gpt4v_input": "textAndImages",
                    }
                },
                "session_state": None,
            },
        )
        time.sleep(5)
        self.client.post(
            "/chat/stream",
            json={
                "messages": [
                    {"content": "Compare the impact of interest rates and GDP in financial markets.", "role": "user"}
                ],
                "context": {
                    "overrides": {
                        "top": 3,
                        "temperature": 0.3,
                        "minimum_reranker_score": 0,
                        "minimum_search_score": 0,
                        "retrieval_mode": "hybrid",
                        "semantic_ranker": True,
                        "semantic_captions": False,
                        "suggest_followup_questions": False,
                        "use_oid_security_filter": False,
                        "use_groups_security_filter": False,
                        "vector_fields": ["embedding", "imageEmbedding"],
                        "use_gpt4v": True,
                        "gpt4v_input": "textAndImages",
                    }
                },
                "session_state": None,
            },
        )
