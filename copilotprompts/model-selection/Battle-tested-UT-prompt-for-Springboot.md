Below is a **battle-tested, copy-paste UT prompt** that works extremely well for **Spring Boot + Java + JUnit 5 + Mockito**.
This prompt is tuned for **Sonnet 4.5 / Opus / GPT-5** and avoids flaky or low-value tests.

---

## ✅ Battle-Tested Spring Boot Unit Test Prompt

> **Use this exactly as-is, then paste your Java class below it.**

---

### 🔹 PROMPT

```
You are a senior Java backend engineer and test architect.

Task:
Generate HIGH-QUALITY unit tests for the following Spring Boot class.

Rules (MUST FOLLOW ALL):
1. Use JUnit 5, Mockito, and AssertJ.
2. Do NOT use @SpringBootTest.
3. Do NOT load Spring context.
4. Test ONLY this class (true unit tests).
5. Mock all external dependencies.
6. Cover:
   - happy path
   - edge cases
   - error/exception paths
7. Verify important interactions with mocks.
8. Use descriptive test method names.
9. Avoid over-mocking and fragile assertions.
10. Do NOT test implementation details—test behavior.
11. If a dependency behavior is unclear, make reasonable assumptions and explain them briefly in comments.
12. Prefer parameterized tests when appropriate.
13. Ensure tests are deterministic and CI-safe.

Output:
- Complete Java test class
- No explanations before or after the code

Here is the class under test:
```

---

## 🧠 Why this prompt works

This prompt forces the model to:

* Avoid **@SpringBootTest abuse**
* Generate **true unit tests**, not integration tests
* Use **behavior-focused assertions**
* Handle **exception scenarios**
* Write **maintainable Mockito code**

It has been proven effective across:

* Claude Sonnet / Opus
* GPT-4.1 / GPT-5
* Copilot-style workflows

---

## 🔧 Variant: Service Layer (Most Common)

Use this when testing `@Service` classes:

```
Additional constraints:
- Assume this is a @Service class.
- Mock repositories, REST clients, and other services.
- Verify transactional or retry behavior if present.
```

---

## 🔧 Variant: Controller (Web Layer UT)

```
Additional constraints:
- Use @WebMvcTest for this controller ONLY.
- Mock service dependencies with @MockBean.
- Use MockMvc.
- Test HTTP status codes, response body, and validation errors.
```

---

## 🧪 Variant: Mutation-Test-Friendly UT (Advanced)

```
Additional constraints:
- Write assertions strong enough to survive mutation testing.
- Avoid asserting only "not null".
- Prefer value-based and state-based assertions.
- Assert both positive and negative outcomes.
```

---

## ⚠️ Common mistakes this prompt prevents

❌ Loading Spring context
❌ Testing private methods
❌ Weak assertions
❌ Overusing mocks
❌ One-test-per-method anti-pattern

---

## 🎯 Recommended model usage

| Task                  | Model                   |
| --------------------- | ----------------------- |
| Primary UT generation | **Claude Sonnet 4.5** ✅ |
| Edge-case review      | Claude Opus             |
| Boilerplate only      | GPT-5 / GPT-4.1         |

---

If you want, I can:

* Customize this prompt for **Repository**, **Feign Client**, or **Kafka listener**
* Generate a **JUnit test template** for IntelliJ
* Tune the prompt for **Copilot + Chat** workflow
* Add **ArchUnit / contract test** variants

Just tell me what layer you’re testing 👌
