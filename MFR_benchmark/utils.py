import pathlib


def get_assets_dir():
    return f'{pathlib.Path(__file__).resolve().parents[0]}/assets'
