Great — here are the **enhanced versions** you asked for, tailored specifically for your workflow as a **Java + Spring Boot senior backend developer using IntelliJ + GitHub Copilot**.

I’m giving you three sets:

1. ✅ **Spring Boot–optimized test update prompts**
2. ✅ **IntelliJ macro-style reusable prompts**
3. ✅ **A clean “Update Class → Update Test” workflow you can use every day**

---

# ✅ 1. **Spring Boot–Optimized Test Update Prompts**

These prompt templates work best when your app uses:
**Spring Boot, JUnit 5, Mockito, MockMvc, WebTestClient, Testcontainers**, etc.

---

### **🧪 A. Update a Service Test (Mockito)**

```
Update the existing test class <ClassName>Test to reflect the updated logic in <ClassName>.
Use JUnit 5 + Mockito.

What to update:
- new branches, validations, or parameters
- updated interactions with injected dependencies
- Mockito verify() expectations
- returned values or exceptions

Do not create a new test file. Modify the existing one only.
Use Arrange / Act / Assert structure.
```

---

### **🌐 B. Update a Controller Test (MockMvc / WebTestClient)**

```
Update the existing controller test <ClassName>Test to match the new request/response logic in <ClassName>.
Use MockMvc (or WebTestClient if applicable).

Please update:
- endpoint paths
- request bodies
- response structures
- HTTP status expectations
- JSON field expectations
- new validation behavior

Modify the existing test ONLY. Do not generate a new class.
```

---

### **📦 C. Update Repository / JPA Tests**

```
Update the existing <RepositoryName>Test to match the updated repository logic.
Apply:
- new query method behavior
- entity field changes
- transactional behavior
- Testcontainers/JPA setup if needed

Patch the existing test file only.
```

---

### **🧰 D. Update Tests When a Bean / Autowired Dependency Changes**

```
Update <ClassName>Test to match the new constructor arguments and dependency behavior introduced in <ClassName>.
Ensure mocks/beans are updated and interactions reflect the new logic.
Modify existing test file only.
```

---

# ✅ 2. **IntelliJ Macro-Style Prompt Templates (Reusable Daily)**

Copy/paste these into your Copilot Chat “favorite prompts” list.

---

### **🎯 Macro 1 — Universal Test Updater**

```
Update the existing test for this class. Adjust mocks, assertions, inputs, outputs, and branches based on the current implementation. Do not create a new test file. Patch only the existing one.
```

---

### **🔬 Macro 2 — Method-Specific Refactor**

```
Update the tests for the method <methodName>() within <ClassName>Test to reflect the updated behavior. Only update affected test cases.
```

---

### **📐 Macro 3 — “Fix my tests after refactor”**

```
Fix and update the existing test class so that all tests pass with the updated implementation. Adjust expectations, mocks, method calls, and validation logic as needed. Do not create new tests unless required by new logic.
```

---

### **🧠 Macro 4 — “Full Coverage Update”**

```
Update the existing test file to provide full branch coverage of the updated class logic. Add missing cases, update assertions, and modify mocks. Do not create a new file.
```

---

### **📄 Macro 5 — “Compare old vs new logic”**

```
Compare the previous version of this class with the current version and update the existing test class accordingly. Keep the structure, modify only what changed.
```

---

# ✅ 3. **Daily Workflow: Update Class → Update Test (Fast & Reliable)**

Here's a clean workflow tailored for IntelliJ + Copilot + Spring Boot.

---

## **1️⃣ Step 1 — Make your code changes**

Modify your class (service, controller, repository, etc.)

---

## **2️⃣ Step 2 — Open your test file side-by-side**

This greatly improves Copilot’s accuracy.

---

## **3️⃣ Step 3 — Use this prompt (the most reliable one)**

```
Update <ClassName>Test to fully match the updated logic in <ClassName>.
Modify the existing test only. Update assertions, mocks, interactions, and test cases.
Use JUnit 5 and Mockito.
```

---

## **4️⃣ Step 4 — Ask Copilot to fix remaining failures**

After running tests:

```
Fix failing tests in <ClassName>Test. Explain what changed and update assertions/mocks accordingly. Modify existing tests only.
```

---

## **5️⃣ Step 5 — (Optional) Ask for edge cases**

```
Add missing edge case tests for <ClassName> based on the updated logic. Do not remove existing tests.
```

---

If you want, I can also create:
✅ a **one-page cheat sheet** (perfect to paste on your IDE wall)
✅ an **IntelliJ Live Template** for these prompts
✅ a **Spring Boot–specific testing style guide** (AAA, given/when/then, mocking best practices)

Would you like any of those?
