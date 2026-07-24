"""
modules/memory/memory.py
━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
- No setup needed — in-memory storage
- Conversation history session मध्ये राहते

HOW TO USE:
   from modules.memory.memory import save_to_memory, get_history
   save_to_memory("question", "answer")
   history = get_history()
"""

from langchain.memory import ConversationBufferMemory

# ── Memory Store (single session) ─────────────────
_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def save_to_memory(user_msg: str, ai_msg: str):
    """
    Conversation history save करतो
    """
    _memory.save_context(
        {"input":  user_msg},
        {"output": ai_msg}
    )

def get_history() -> list:
    """
    मागील conversations return करतो
    """
    return _memory.load_memory_variables({}).get(
        "chat_history", []
    )

def clear_memory():
    """
    Memory clear करतो (new session साठी)
    """
    _memory.clear()
