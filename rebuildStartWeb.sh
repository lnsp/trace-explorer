pip uninstall -y trace_explorer
python setup.py bdist_wheel
pip install ./dist/trace_explorer-*
trace_explorer web
