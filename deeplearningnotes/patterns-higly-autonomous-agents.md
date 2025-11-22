

## 🤖 Planning with Code Execution: The Core Idea

The lecture argues that for complex tasks, it's generally **better** to let the LLM **write and execute code** (like Python or SQL) than to provide it with a small set of **custom tools**.

---

### Key Contrast

| Method | What the LLM Uses | Why It's Better |
| :--- | :--- | :--- |
| **Custom Tools** (Less Effective) | A few **fixed tools** written by a developer. | Toolset is **too limited** and requires constant maintenance for new queries. |
| **Code as Action** (More Effective) | **Full programming language** (e.g., Python/Pandas) and its vast libraries. | Provides a **massive, flexible toolset** that the LLM is already highly proficient in, enabling it to write rich, dynamic plans. |

**In short: Stop writing custom tools; start letting the LLM write the code.** 