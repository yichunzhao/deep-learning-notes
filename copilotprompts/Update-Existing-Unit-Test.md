Great — here are **ready-to-use prompt templates** specifically for **IntelliJ + GitHub Copilot**, designed to reliably update unit tests after you modify a class.

You can copy/paste these directly into Copilot Chat.

---

# 🎯 **1. General “Update Existing Unit Test” Template**

Use this when you changed the class and want the test updated.

**Prompt:**

```
Update the existing unit test file for <ClassName> so it matches the updated implementation.
Keep the existing test structure but adjust:
- assertions
- mocked behavior
- test inputs/outputs
- edge cases
Only modify the existing test class, do NOT create a new one.
```

---

# 🧩 **2. Template When You Are Inside the Updated Class**

Use from the main class file.

**Prompt:**

```
Based on the updated logic in this class, rewrite the existing test file <ClassName>Test to reflect the new behavior.
Modify only the existing test file. Update assertions, mocks, and test cases as needed.
Use JUnit 5 and Mockito.
```

---

# 🧭 **3. Template When Copilot Keeps Creating a New Test Class**

Use this to force Copilot to patch your current test file.

**Prompt:**

```
Do not generate a new test class. Update ONLY the existing test:
<path to test>/ <ClassName>Test.java

Modify:
- test cases
- assertions
- mocks
to match the new logic in <ClassName>.
Keep file structure and naming conventions the same.
```

---

# 🧪 **4. Template for Updating Specific Method Tests**

Use after modifying a single method.

**Prompt:**

```
Update the unit tests related to the method <methodName>() in <ClassName>Test.
Reflect changes in:
- return values
- exception handling
- validation logic
- branches / edge cases
Only modify the affected tests.
```

---

# 🧱 **5. Template for More Precise Test Behavior**

Use for complex classes or services.

**Prompt:**

```
Analyze the updated <ClassName> implementation and update the unit test accordingly.

Ensure the test covers:
- all branches of the updated logic
- new error conditions or exceptions
- new parameters or fields
- new interactions with dependencies

Keep JUnit 5, Mockito, and AAA (Arrange/Act/Assert) structure. Modify the existing test only.
```

---

# 📘 **6. Template When You Want Copilot to Compare Old vs. New Logic**

(This works well if you didn’t make huge changes.)

**Prompt:**

```
Compare the previous logic in <ClassName> with the updated code and update the unit test to match:
- change assertions where needed
- update mocks and interactions
- revise edge cases
- add/remove test cases as required

Do not create a new test file—patch the existing one only.
```

---

# ⚡ Bonus: A One-Line Power Prompt

If you want something short and effective:

**Prompt:**

```
Update <ClassName>Test to fully match the new logic in <ClassName>, modifying only the existing test file.
```

---

If you'd like, I can also give you:
✅ a version customized for Java + Spring Boot
✅ an IntelliJ macro-style prompt
✅ a workflow for “update class → update test” automation with Copilot

Want those?


