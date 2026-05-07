# -*- mode: python ; coding: utf-8 -*-

import sys, os
from pathlib import Path

# spaCy is a core feature — always bundled with the en_core_web_sm model.

def _find_spacy_model_dir():
    """Find en_core_web_sm model installation path for bundling into exe."""
    try:
        import en_core_web_sm
        return os.path.dirname(en_core_web_sm.__file__)
    except ImportError:
        site_packages = Path(sys.executable).parent / 'Lib' / 'site-packages'
        model_dir = site_packages / 'en_core_web_sm'
        if model_dir.is_dir():
            return str(model_dir)
        return None

_spacy_dir = _find_spacy_model_dir()

_datas = [('src/srt_translator', 'srt_translator')]
_hiddenimports = [
    'srt_translator',
    'srt_translator.config', 'srt_translator.models', 'srt_translator.parser',
    'srt_translator.merger', 'srt_translator.glossary', 'srt_translator.llm_client',
    'srt_translator.pipeline', 'srt_translator.translator', 'srt_translator.text_utils',
    'srt_translator.progress',
    'spacy', 'spacy.lang.en', 'spacy.cli',
    'en_core_web_sm', 'thinc',
]

if _spacy_dir:
    _datas.append((_spacy_dir, 'en_core_web_sm'))
    print(f"INFO: Bundling spaCy model from: {_spacy_dir}")
else:
    print("WARNING: en_core_web_sm model not found. Install: python -m spacy download en_core_web_sm")


a = Analysis(
    ['api_server.py'],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='api-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
