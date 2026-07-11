# Data sources and normalized contract

The build command converts each supported source to the same internal message shape before it creates reply examples.

## Normalized message fields

| Field | Type | Meaning |
|---|---|---|
| `conversation_id` | string | Pseudonymous conversation identifier |
| `sender_role` | `me` or `other` | Whether the archive owner authored the message |
| `timestamp` | integer | Unix time in seconds |
| `text` | string | Normalized message text |
| `source` | string | `imessage`, `x_dm`, `x_reply`, or `gmail` |
| `medium` | string | `text`, `dm`, `public`, or `email` |
| `context_hint` | optional string | Extra context such as an email subject |

## iMessage JSONL

The iMessage loader consumes the pseudonymized message output from [`texts-to-transformer`](https://github.com/Doriandarko/texts-to-transformer). Each line is one JSON object:

```json
{"chat_id":"chat_demo_1","timestamp_ns":1700000000000000000,"sender_role":"other","text":"Could we move this to Thursday?"}
```

Required fields are `chat_id`, `timestamp_ns`, `sender_role`, and `text`. `timestamp_ns` is Unix time in nanoseconds. Empty text and unknown sender roles are skipped.

The upstream extractor handles Apple attributed-body text, reaction and system filtering, attachment-only rows, session preparation, and keyed pseudonyms. This release was tested with commit `90419977f4b0ab80c7c160faa2adcbcc2bfe7943`. Pin the upstream commit you audit rather than assuming every future schema is compatible.

## X archive

Request an official archive from X account settings and pass the downloaded ZIP directly to `--x-archive` after moving it to a private directory outside Git.

The loader reads:

- `account.js` to identify the archive owner
- `direct-messages.js` and `direct-messages-group.js` for private conversations
- `tweets.js` for authored public replies

It does not call the X API. Repeated long outreach templates are filtered before reply pairing.

## Gmail JSON

The Gmail loader accepts decoded JSON containing either:

- a top-level `threads` array with `messages` inside each thread
- a single top-level `thread`
- a top-level `messages` page, as returned by `gog gmail messages search --json`
- a successful message-page response under `data.messages`

Messages may use Gmail API `payload.headers` and MIME bodies or decoded convenience fields such as `body`, `text`, `decodedBody`, or `decoded_body`. HTML is reduced to visible text. Quoted replies, forwarded sections, common mobile signatures, and attachment-only placeholders are removed.

Writer addresses determine `me` versus `other`. Put one address per line in a mode-`0600` file and pass it through `--gmail-owner-file`.

## Pairing behavior

Messages are grouped by pseudonymous conversation. A gap longer than six hours starts a new session. Consecutive messages from the same sender within two minutes merge into one turn.

Every `me` turn immediately following an `other` turn becomes a contextual example. Authored standalone messages are also retained as style evidence when no prompt can be recovered.
