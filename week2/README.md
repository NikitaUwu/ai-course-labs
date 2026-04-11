# Week 2 setup

If `py -m pip install -r requirements.txt` fails inside `week2`, first check the shared virtual environment in `ai-course-labs\venv`.

In this workspace the existing `venv` is broken: its `pyvenv.cfg` points to
`C:\Users\nikit\AppData\Local\Programs\Python\Python313\python.exe`, but that
interpreter is no longer available. In that state even `venv\Scripts\python.exe`
cannot start correctly.

Recommended setup steps:

```powershell
cd C:\Users\nikit\Desktop\ML\repo\ai-course-labs
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
cd .\week2
python -m pip install -r requirements.txt
```

If `py -3.13` is unavailable on your machine, use the full path to the Python
interpreter that you actually have installed.

For new code prefer:

```python
from langchain_chroma import Chroma
```

The older community import may still exist, but the dedicated `langchain-chroma`
package is the safer option for a fresh lab.
