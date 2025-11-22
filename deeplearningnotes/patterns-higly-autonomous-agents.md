

## 🤖 Planning with Code Execution: The Core Idea

The lecture argues that for complex tasks, it's generally **better** to let the LLM **write and execute code** (like Python or SQL) than to provide it with a small set of **custom tools**.

---

### Key Contrast

| Method | What the LLM Uses | Why It's Better |
| :--- | :--- | :--- |
| **Custom Tools** (Less Effective) | A few **fixed tools** written by a developer. | Toolset is **too limited** and requires constant maintenance for new queries. |
| **Code as Action** (More Effective) | **Full programming language** (e.g., Python/Pandas) and its vast libraries. | Provides a **massive, flexible toolset** that the LLM is already highly proficient in, enabling it to write rich, dynamic plans. |

**In short: Stop writing custom tools; start letting the LLM write the code.** 

## 📝 Multi-Agent Systems

This lecture introduced the concept of **Multi-Agent Systems** as a powerful way to handle **complex tasks** by having multiple specialized agents (each powered by an LLM) collaborate.

---

### 1. 💡 Why Use Multiple Agents? (The Analogy)

* **Problem:** For a complex task, building one single LLM agent to do everything can be difficult to manage and prone to errors.
* **Analogy:** We break down complex work into:
    * Multiple **processes/threads** on a single computer.
    * A **team of people** with different roles (e.g., a researcher, designer, writer) instead of just one generalist.
* **Benefit:** This approach makes it easier for developers to **decompose a complex task** into smaller, manageable sub-tasks that can be addressed by specialized agents. 

---

### 2. 🤹 Creating Specialized Agents (Roles and Tools)

An agent is created by **prompting an LLM** to play a specific role. Each agent is then given the **tools** it needs for its job.

| Agent Role | Example Task | Key Tool(s) Needed |
| :--- | :--- | :--- |
| **Researcher** | Analyze market trends and competitor offerings. | **Web Search** |
| **Graphic Designer** | Create visualizations, charts, and artwork. | **Image Generation/Manipulation APIs**, **Code Execution** (for charts) |
| **Writer** | Transform research into report text/marketing copy. | **No specific tools** (uses LLM's core text generation ability) |
| **Marketing Manager** | Coordinate and review the work of the team. | **(Acts as an Orchestrator)** |

---

### 3. 🗺️ Communication Patterns (Workflows)

The lecture showed two ways agents can work together, which defines the **communication pattern** or workflow:

* **1. Linear Workflow (Pipeline):**
    * The simplest pattern where one agent's output immediately becomes the next agent's input.
    * **Example:** Research Agent $\rightarrow$ Graphic Designer Agent $\rightarrow$ Writer Agent $\rightarrow$ Final Report.
    * **Advantage:** Simple to design and debug.
* **2. Manager/Coordinator Workflow (Delegation):**
    * A **manager agent** (the LLM) is at the top. It receives the complex request, determines a plan, and delegates specific tasks to the worker agents.
    * The workers report back to the manager.
    * **Example:** Marketing Manager Agent plans: 1. Ask researcher. 2. Ask designer. 3. Ask writer. 4. **Review/Improve final report.** (The manager acts as a fourth agent).
    * **Advantage:** Allows for dynamic planning, reflection, and final quality control by the managing agent.

---

### ✅ Conclusion

Building a multi-agent system allows you to build **reusable** and **highly specialized** agents that collectively solve complex problems more effectively than a single, generalized agent. Designing the right **communication pattern** is key to success.
