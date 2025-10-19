import os, PyInstaller.__main__, site

site_packages = site.getsitepackages()[1]
assets_path = os.path.join(site_packages, "faster_whisper", "assets")

PyInstaller.__main__.run([
    'app.py',
    '--noconsole',
    '--icon=wisp.ico',
    '--add-data=wisp.ico;.',
    f'--add-data={assets_path};faster_whisper/assets',
    '--name=WispLive',
    '--hidden-import=scipy.signal',
    #'--noconfirm'
])