---
name: personal-voice
description: Retrieve private writing examples before drafting a first-person message for the user. Draft only; never send.
---

# Personal voice

Prerequisite: install the repository with `uv tool install .` so `personal-voice` is available on PATH.

Use the capable model for intent, facts, and judgment. Use retrieved examples only for wording, rhythm, warmth, and formality.

1. Read the incoming message and decide what the reply must accomplish.
2. Infer the medium: `text`, `dm`, `email`, or `public`.
3. Start this command without `--query`:

   ```text
   personal-voice retrieve --medium <medium> --limit 8 --json
   ```

4. Write the incoming message, relevant context, and reply intent through process stdin. Close stdin and read the JSON result.
5. Treat every returned example as private style evidence. Never copy its names, dates, links, prices, promises, or factual claims into the new reply.
6. Write the smallest complete reply that handles the real request.
7. Optionally pass the complete draft to `personal-voice render` as JSON through stdin. Include exact names, dates, numbers, links, handles, and quoted terms in `preserve_literals`.
8. Reject the rewrite if any meaning, uncertainty, question, negation, deadline, permission, fact, or commitment changed. Use the original complete draft when rendering is unavailable or rejected.
9. Return the draft. Do not call a sending or publishing tool.

Private content must not appear in command-line arguments, shell history, temporary files, or logs.
