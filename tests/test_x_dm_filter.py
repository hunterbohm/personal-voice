from personal_voice.x_dm_filter import filter_automated_x_dms


def message(conversation: str, role: str, text: str, timestamp: int) -> dict:
    return {
        "conversation_id": conversation,
        "sender_role": role,
        "timestamp": timestamp,
        "text": text,
        "source": "x_dm",
        "medium": "dm",
    }


def test_filter_removes_long_templates_repeated_across_five_conversations() -> None:
    broadcast = (
        "Hey! I wanted to reach out because we help founders automate their entire "
        "outbound pipeline. Would you be open to a quick call this week?"
    )
    messages = [message(f"chat-{index}", "me", broadcast, index) for index in range(1, 6)]
    messages.append(message("chat-human", "me", "yeah I can take a look tomorrow", 6))

    retained = filter_automated_x_dms(messages)

    assert [row["timestamp"] for row in retained] == [6]


def test_filter_removes_personalized_templates_in_both_directions() -> None:
    messages = []
    for index, name in enumerate(("Alice", "Bob", "Casey", "Drew", "Evan"), start=1):
        messages.append(
            message(
                f"outbound-{index}",
                "me",
                f"Hey {name}, we help small teams automate repetitive operations without "
                f"hiring more people. See https://example.com/{index} and message "
                f"@contact{index} if you want the details for cohort {index}.",
                index,
            )
        )
        messages.append(
            message(
                f"inbound-{index}",
                "other",
                f"Hello {name}, thanks for connecting with our account. Here is your custom "
                f"welcome guide https://example.org/{index}, reference number {index}. Let us "
                "know if your team wants more information.",
                index + 10,
            )
        )

    retained = filter_automated_x_dms(messages)

    assert retained == []


def test_filter_preserves_short_replies_even_when_they_repeat_often() -> None:
    messages = [
        message(f"chat-{index}", "me", "yeah sounds good to me", index) for index in range(1, 21)
    ]

    retained = filter_automated_x_dms(messages)

    assert retained == messages


def test_filter_requires_five_distinct_conversations() -> None:
    template = (
        "Hello there, this deliberately long repeated message has enough words to qualify "
        "for filtering only when it appears across five separate conversations."
    )
    messages = [message("same-chat", "me", template, index) for index in range(1, 6)]
    messages.extend(
        message(f"different-{index}", "me", template, index + 10) for index in range(1, 4)
    )

    retained = filter_automated_x_dms(messages)

    assert retained == messages


def test_filter_does_not_touch_non_x_messages() -> None:
    template = (
        "Hello there, this is a long repeated email template that appears many times but "
        "must remain because this filter is only responsible for direct messages from X."
    )
    messages = [
        {**message(f"chat-{index}", "me", template, index), "source": "gmail"}
        for index in range(1, 6)
    ]

    retained = filter_automated_x_dms(messages)

    assert retained == messages
