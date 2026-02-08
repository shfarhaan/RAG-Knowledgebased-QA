# Virtual Environment Setup Guide

## Why Use a Virtual Environment?

A virtual environment is an isolated Python environment that:
- ✅ Keeps project dependencies separate from system Python
- ✅ Prevents conflicts between different projects
- ✅ Makes your project portable and reproducible
- ✅ Allows different Python versions per project
- ✅ Prevents accidentally breaking system Python packages

**Best Practice**: Always use a virtual environment for Python projects!

---

## 🚀 Quick Setup (Recommended)

### Windows - Automatic Setup
```bash
# Just run this - it creates venv AND installs everything!
install.bat
```

That's it! The script will:
1. Create a virtual environment in `venv/` folder
2. Activate it automatically
3. Install all dependencies

---

## 📋 Manual Setup

### Windows - PowerShell

```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
.\venv\Scripts\Activate.ps1

# 3. You should see (venv) in your prompt:
(venv) PS F:\Selise Assessment>

# 4. Install dependencies
pip install -r requirements.txt
```

### Windows - Command Prompt

```cmd
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate.bat

# 3. You should see (venv) in your prompt:
(venv) F:\Selise Assessment>

# 4. Install dependencies
pip install -r requirements.txt
```

### Mac/Linux - Bash/Zsh

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. You should see (venv) in your prompt:
(venv) user@computer:~/Selise Assessment$

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ✅ Verify Virtual Environment is Active

### Check 1: Prompt Indicator
You should see `(venv)` at the start of your command prompt:
```
(venv) F:\Selise Assessment>
```

### Check 2: Python Location
```bash
# Windows
where python

# Mac/Linux
which python
```

Should show path inside your project's `venv` folder:
```
F:\Selise Assessment\venv\Scripts\python.exe  ✅ Correct!
C:\Python310\python.exe  ❌ Wrong - using system Python
```

### Check 3: Run Setup Script
```bash
python setup.py
```

Will tell you if you're in a virtual environment or not.

---

## 🔄 Using the Virtual Environment

### Every Time You Open a New Terminal

You need to **reactivate** the virtual environment:

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Running the Application

Always activate venv first, then run:
```bash
# Activate venv (if not already active)
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Then run the application
streamlit run app.py

# Or CLI
python cli.py chat

# Or notebook
jupyter notebook
```

### Deactivating

When you're done working:
```bash
deactivate
```

---

## 🎯 VS Code Integration

### Automatic Activation in VS Code

1. **Open Command Palette**: `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
2. **Type**: "Python: Select Interpreter"
3. **Choose**: `.\venv\Scripts\python.exe` (the one in your project folder)

Now VS Code will:
- ✅ Automatically activate venv in integrated terminal
- ✅ Use venv for Python IntelliSense
- ✅ Resolve import warnings correctly

### Settings for VS Code

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true
}
```

---

## 📦 Installing New Packages

When you're in the virtual environment:

```bash
# Install a single package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt
```

---

## 🗑️ Removing Virtual Environment

If you need to start fresh:

**Windows:**
```powershell
# Deactivate first
deactivate

# Delete the folder
Remove-Item -Recurse -Force venv

# Create new one
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
# Deactivate first
deactivate

# Delete the folder
rm -rf venv

# Create new one
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ❓ Common Issues

### Issue: "Execution of scripts is disabled on this system"

**Windows PowerShell only** - You need to allow script execution:

```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

### Issue: Import warnings still showing in VS Code

1. Make sure venv is activated: `.\venv\Scripts\Activate.ps1`
2. Select correct Python interpreter in VS Code (Ctrl+Shift+P → Python: Select Interpreter)
3. Reload VS Code window

### Issue: "pip not found" or "command not found"

Make sure virtual environment is activated - you should see `(venv)` in your prompt.

### Issue: Packages installed but imports still fail

1. Verify you're in virtual environment: `where python` (Windows) or `which python` (Mac/Linux)
2. Check packages are installed: `pip list`
3. Restart your terminal/IDE

---

## 📊 Quick Reference

| Action | Windows (PowerShell) | Mac/Linux |
|--------|---------------------|-----------|
| Create venv | `python -m venv venv` | `python3 -m venv venv` |
| Activate | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Deactivate | `deactivate` | `deactivate` |
| Check active | `where python` | `which python` |
| Install packages | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Run app | `streamlit run app.py` | `streamlit run app.py` |

---

## 🎓 Understanding Virtual Environments

### What Gets Isolated?

✅ **Isolated:**
- Python packages (installed via pip)
- Package versions
- Python interpreter (copy or symlink)

❌ **Not Isolated:**
- Your project code
- System environment variables
- Global Python installation

### Where Are Packages Installed?

**Without venv:**
```
C:\Users\YourName\AppData\Local\Programs\Python\Python310\Lib\site-packages\
```

**With venv:**
```
F:\Selise Assessment\venv\Lib\site-packages\
```

This keeps everything contained in your project folder!

---

## ✅ Best Practices

1. **Always create venv BEFORE installing packages**
   ```bash
   python -m venv venv  # Create
   .\venv\Scripts\Activate.ps1  # Activate
   pip install -r requirements.txt  # Install
   ```

2. **Never commit venv folder to git**
   - Add `venv/` to `.gitignore`
   - Share `requirements.txt` instead

3. **Use requirements.txt to share dependencies**
   ```bash
   # Export current packages
   pip freeze > requirements.txt
   
   # Others can recreate environment
   pip install -r requirements.txt
   ```

4. **Reactivate venv in each new terminal session**

5. **Use VS Code Python interpreter selector** for automatic activation

---

## 🚀 You're All Set!

Your virtual environment is now set up! Remember:

1. **Always activate** before working: `.\venv\Scripts\Activate.ps1`
2. Look for **(venv)** in your prompt
3. Run `python setup.py` to verify everything

**Start the application:**
```bash
# Make sure venv is active first!
streamlit run app.py
```

Happy coding! 🎉
