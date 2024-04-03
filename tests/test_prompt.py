from text2graph.prompt import PromptHandlerV3


def test_prompt_handler_v3(text):
    handler = PromptHandlerV3()

    sys_prompt = handler.get_system_prompt(text)
    assert sys_prompt is not None

    user_prompt = handler.get_user_prompt(text)
    assert user_prompt is not None

    messages = handler.get_gpt_messages(text)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == sys_prompt
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == user_prompt
