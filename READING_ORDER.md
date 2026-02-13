# Reading Order Guide - Understand This Project

## The Learning Path (Start Here!)

### 🟢 Level 1: "What Is This?" (5 min read)

**Start with:** `README.md`

**What you'll learn:**
- High-level overview of the project
- What MOF generation is
- Architecture diagram (quantum + classical)
- Key features checklist
- File structure

**After reading, you'll know:**
- ✓ This is a quantum-classical MOF generator
- ✓ It uses 40 qubits + neural networks
- ✓ It generates material structures
- ✓ What files do what

---

### 🟡 Level 2: "How Do I Use It?" (10 min read)

**Next:** `HPC_DEPLOYMENT_GUIDE.md` → Section: "When You Get to the HPC"

**What you'll learn:**
- Exact commands to run on HPC
- Step-by-step walkthrough
- Copy/paste snippets
- Typical workflow timeline

**After reading, you'll know:**
- ✓ How to login to HPC
- ✓ How to setup Python environment
- ✓ How to generate MOF structures
- ✓ How to submit batch jobs
- ✓ Where your results will be

**Read this when:**
- You're about to deploy on actual HPC
- You want to know "what command do I run?"

---

### 🟠 Level 3: "What Does Each Component Do?" (30 min read)

**Then:** `HPC_DEPLOYMENT_GUIDE.md` → Section: "Component-by-Component Breakdown"

**What you'll learn:**
- Quantum Circuit: 3 layers explained
- Readout Mapper: How measurements become coordinates
- Generator: How everything orchestrates
- Why each part exists

**After reading, you'll know:**
- ✓ What Hadamard gates do
- ✓ Why StronglyEntanglingLayers are used
- ✓ How quantum outputs map to atoms
- ✓ Why classical MLP is needed
- ✓ Data flow: noise → qubits → atoms

**Read this when:**
- You want to understand the architecture in depth
- You're debugging something not working
- You want to modify hyperparameters

---

### 🔴 Level 4: "Show Me Every Line of Code" (60 min read)

**Finally:** `HPC_DEPLOYMENT_GUIDE.md` → Section: "File-by-File Deep Dive"

**What you'll learn:**
- Every class and method explained
- Every parameter documented
- Input shapes and output shapes
- Code examples with annotations

**Coverage:**
- `hybrid_circuit.py` - Quantum circuit implementation
- `hybrid_mapper.py` - Classical decoder
- `qgan_generator.py` - End-to-end orchestration

**After reading, you'll know:**
- ✓ Every line of code and why it exists
- ✓ Tensor shapes at each step
- ✓ Parameter meanings
- ✓ How to modify/extend the code

**Read this when:**
- You need to modify the code
- You're teaching someone else
- You're debugging a specific issue

---

## Document Map

```
README.md
├─ What is this project?
├─ Architecture overview
├─ Feature checklist
└─ File structure

IMPLEMENTATION_STATUS.md
├─ What was built
├─ Verification results
├─ Key insights
└─ Next steps

HPC_DEPLOYMENT_GUIDE.md
├─ When You Get to HPC
│  └─ Exact commands to run (START HERE for deployment)
├─ System Architecture Overview
│  └─ Big picture + why quantum vs classical
├─ Component-by-Component Breakdown
│  └─ Quantum circuit, mapper, generator explained
└─ File-by-File Deep Dive
   └─ Every method with full annotations

.github/copilot-instructions.md
├─ Code style guidelines
├─ Architecture patterns
└─ Project conventions
```

---

## Reading Paths by Use Case

### Use Case 1: "I Just Want to Run It"

```
1. HPC_DEPLOYMENT_GUIDE.md 
   → Section: "When You Get to the HPC"
   (5 min)

2. quick_test.py
   (Run it, see it work)

Done! You can generate MOFs now.
```

---

### Use Case 2: "I Need to Understand the Architecture"

```
1. README.md
   (5 min) → Understand what this is

2. HPC_DEPLOYMENT_GUIDE.md
   → "System Architecture Overview"
   (10 min) → Understand why it's designed this way

3. HPC_DEPLOYMENT_GUIDE.md
   → "Component-by-Component Breakdown"
   (30 min) → Understand each part in detail

Done! You understand the full architecture.
```

---

### Use Case 3: "I Need to Modify the Code"

```
1. README.md
   (5 min) → Context

2. HPC_DEPLOYMENT_GUIDE.md
   → "System Architecture Overview"
   (10 min) → Understand the design

3. Look at the relevant source file
   (hybrid_circuit.py OR hybrid_mapper.py OR qgan_generator.py)

4. HPC_DEPLOYMENT_GUIDE.md
   → "File-by-File Deep Dive"
   → Find your file's section
   (20 min) → Every line explained

5. Make your changes

Done! You know what you're changing and why.
```

---

### Use Case 4: "I Need to Debug Something"

```
1. HPC_DEPLOYMENT_GUIDE.md
   → "Understanding Outputs"
   (10 min) → What should results look like?

2. Run: python3 quick_test.py
   → Does it pass?

3. If test fails, check: HPC_DEPLOYMENT_GUIDE.md
   → "File-by-File Deep Dive"
   → Find the failing component

4. Read the specific method's explanation
   (understand inputs/outputs)

5. Add debug prints or check tensor shapes

Done! You found the bug.
```

---

### Use Case 5: "I'm Deploying to Real Quantum Hardware"

```
1. README.md
   (5 min) → Overview

2. HPC_DEPLOYMENT_GUIDE.md
   → "What to Run on HPC"
   → export_qasm() method
   (10 min) → Export circuit to QASM

3. HPC_DEPLOYMENT_GUIDE.md
   → "File-by-File Deep Dive"
   → QGAN_Generator.export_qasm()
   (15 min) → Understand QASM export

4. Use circuit.qasm with Fujitsu Quantum Simulator

Done! You have circuit ready for real hardware.
```

---

## Quick Reference: Find What You Need

| Question | Document | Section |
|----------|----------|---------|
| What is a MOF? | README.md | Architecture Overview |
| How do I run this? | HPC_DEPLOYMENT_GUIDE.md | When You Get to HPC |
| What does this code do? | HPC_DEPLOYMENT_GUIDE.md | Component-by-Component |
| What's inside this function? | HPC_DEPLOYMENT_GUIDE.md | File-by-File Deep Dive |
| What should my output look like? | HPC_DEPLOYMENT_GUIDE.md | Understanding Outputs |
| How do I debug this? | HPC_DEPLOYMENT_GUIDE.md | Understanding Outputs + Component-by-Component |
| What code style should I follow? | .github/copilot-instructions.md | Code Style |
| What was implemented? | IMPLEMENTATION_STATUS.md | Checklist |
| How do I export the circuit? | HPC_DEPLOYMENT_GUIDE.md | export_qasm() method |

---

## Detailed Reading Order (Complete Path)

**If you're starting from zero and want full understanding:**

### Week 1: Foundation
- **Monday**: Read `README.md` (5 min)
  - Understand what MOF generation is
  - See architecture diagram
  
- **Tuesday**: Read `IMPLEMENTATION_STATUS.md` (10 min)
  - Know what was built
  - Understand verification results
  
- **Wednesday**: Read `HPC_DEPLOYMENT_GUIDE.md` → "System Architecture Overview" (15 min)
  - Understand why quantum+classical
  - See data flow

### Week 2: Hands-On
- **Thursday**: Run `python3 quick_test.py` (5 min)
  - See it work
  - Understand typical output
  
- **Friday**: Read `HPC_DEPLOYMENT_GUIDE.md` → "When You Get to HPC" (10 min)
  - Know exact commands
  - Prepare for deployment

### Week 3: Deep Dive
- **Monday**: Read `HPC_DEPLOYMENT_GUIDE.md` → "Component-by-Component Breakdown" (30 min)
  - Understand each quantum layer
  - Understand classical mapping
  
- **Tuesday-Wednesday**: Read `HPC_DEPLOYMENT_GUIDE.md` → "File-by-File Deep Dive" (60 min)
  - Every method annotated
  - Every tensor shape documented
  
- **Thursday**: Review `.github/copilot-instructions.md` (10 min)
  - Code conventions
  - Style guidelines

- **Friday**: Modify the code (now you know what you're changing!)

---

## For Different Audiences

### For PIs / Project Managers
**"I need a 10-minute overview"**
1. README.md → Architecture section
2. IMPLEMENTATION_STATUS.md → Checklist
3. Done! ✓

### For HPC Systems Administrators
**"I need to set this up on our cluster"**
1. README.md → Requirements section
2. HPC_DEPLOYMENT_GUIDE.md → When You Get to HPC
3. Done! ✓

### For Quantum Physicists
**"Show me the quantum circuit"**
1. HPC_DEPLOYMENT_GUIDE.md → Component-by-Component → Quantum Circuit
2. hybrid_circuit.py (source code)
3. Done! ✓

### For Machine Learning Engineers
**"Show me the neural networks"**
1. HPC_DEPLOYMENT_GUIDE.md → Component-by-Component → Readout Mapper
2. hybrid_mapper.py (source code)
3. Done! ✓

### For Software Engineers
**"Show me the architecture"**
1. README.md → Architecture
2. HPC_DEPLOYMENT_GUIDE.md → System Architecture Overview
3. HPC_DEPLOYMENT_GUIDE.md → File-by-File Deep Dive
4. Source code (qgan_generator.py, hybrid_circuit.py, hybrid_mapper.py)
5. Done! ✓

### For Chemistry Students
**"What does this generate?"**
1. README.md → What is a MOF?
2. HPC_DEPLOYMENT_GUIDE.md → Understanding Outputs
3. Done! ✓

---

## TL;DR - Just Get Started

**If you have 15 minutes:**
```
README.md (5 min)
     ↓
HPC_DEPLOYMENT_GUIDE.md - "When You Get to HPC" (10 min)
     ↓
Run: python3 quick_test.py
     ↓
Done! You know what this is and how to use it.
```

**If you have 1 hour:**
```
README.md (5 min)
     ↓
HPC_DEPLOYMENT_GUIDE.md - "System Architecture Overview" (15 min)
     ↓
HPC_DEPLOYMENT_GUIDE.md - "Component-by-Component Breakdown" (30 min)
     ↓
Run: python3 quick_test.py and inspect output
     ↓
Done! You understand the full system.
```

**If you have 3 hours:**
```
README.md (5 min)
     ↓
HPC_DEPLOYMENT_GUIDE.md - "System Architecture Overview" (15 min)
     ↓
HPC_DEPLOYMENT_GUIDE.md - "Component-by-Component Breakdown" (30 min)
     ↓
HPC_DEPLOYMENT_GUIDE.md - "File-by-File Deep Dive" (60 min)
     ↓
.github/copilot-instructions.md (10 min)
     ↓
Run quick_test.py, generate_mofs.py, and inspect code
     ↓
Done! You're ready to modify and deploy.
```

---

## Next Steps After Reading

| What You Want | What to Do |
|---------------|-----------|
| Generate MOFs | `python3 quick_test.py`, then `python3 generate_mofs.py` |
| Deploy to HPC | Follow "When You Get to HPC" section exactly |
| Modify circuit | Read File-by-File section on hybrid_circuit.py, then edit |
| Modify mapper | Read File-by-File section on hybrid_mapper.py, then edit |
| Export for quantum hardware | Run `generator.export_qasm('circuit.qasm')` |
| Train/optimize | Follow example_usage.py → Example 6 |
| Validate results | Run `quick_test.py` and read "Understanding Outputs" |

---

**Start with README.md. Then follow the appropriate use case above.** 🚀
