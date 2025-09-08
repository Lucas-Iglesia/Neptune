# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app/neptune_app.py'],
    pathex=['.'],
    binaries=[],
    # include the model file inside the frozen app under a "model" folder
    datas=[('app/model/nwd-v2.pt', 'model')],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='neptune_app',
    debug=False,
    strip=False,
    upx=True,
    console=False,  # set True if you want a console app
)