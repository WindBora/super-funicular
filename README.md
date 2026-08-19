# Electromagnetic strip-scattering code

The publication workflow uses Python 3.13.1, recorded in `.python-version`,
and exact package versions pinned in `requirements-publication.txt`. From a
fresh clone, create the local environment, verify its interpreter, install the
numerical stack, and regenerate every paper figure, CSV value, LaTeX macro,
and reproducibility manifest with:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:3] == (3, 13, 1), sys.version"
.\.venv\Scripts\python.exe -m pip install -r .\requirements-publication.txt
.\.venv\Scripts\python.exe .\ukraine_microwave_week\generate_figures.py --build
```

The generator validates numerical convergence before compiling
`ukraine_microwave_week/main.tex`. Detailed physical parameters, solver orders,
data fields, and acceptance thresholds are documented in
`ukraine_microwave_week/NOTES.md`.
