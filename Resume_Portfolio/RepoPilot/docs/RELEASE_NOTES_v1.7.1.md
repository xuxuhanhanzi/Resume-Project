# RepoPilot v1.7.1

## P2 follow-up: Async terminal compatibility

- Fixed the Prompt Toolkit integration used by interactive TTY sessions. RepoPilot's CLI already
  owns an asyncio event loop, so its terminal adapter now awaits ``PromptSession.prompt_async()``
  instead of calling synchronous ``prompt()``, which attempted to start a nested event loop.
- The portable synchronous input fallback is retained for non-interactive and embedding callers.
- Added a regression test covering the asynchronous prompt adapter, preventing the
  ``asyncio.run() cannot be called from a running event loop`` startup failure from returning.
