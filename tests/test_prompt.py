def test_strat_prompt_handler_v3(text, stratname_prompt_handler_v3):
    handler = stratname_prompt_handler_v3

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


def test_mineral_prompt_handler_v0(text, mineral_prompt_handler_v0):
    handler = mineral_prompt_handler_v0

    sys_prompt = handler.get_system_prompt(text)
    assert sys_prompt is not None
    assert "gallium" in sys_prompt

    user_prompt = handler.get_user_prompt(text)
    assert user_prompt is not None

    messages = handler.get_gpt_messages(text)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == sys_prompt
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == user_prompt
