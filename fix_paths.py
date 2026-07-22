import re
from pathlib import Path

def process_file(filepath):
    file_path = Path(filepath)
    if not file_path.exists():
        print(f"File not found: {filepath}")
        return
    try:
        content = file_path.read_text(encoding='utf-8')
        encoding = 'utf-8'
    except UnicodeDecodeError:
        content = file_path.read_text(encoding='utf-16')
        encoding = 'utf-16'

    # Replace best_*.pth -> models/weights/best_*.pth
    content = re.sub(r'\'(best_[^\']*?\.pth)\'', r"'models/weights/\1'", content)
    content = re.sub(r'\"(best_[^\']*?\.pth)\"', r'"models/weights/\1"', content)

    # Replace .pkl configs -> models/configs/
    content = re.sub(r'\'([^\']*?\.pkl)\'', r"'models/configs/\1'", content)
    content = re.sub(r'\"([^\']*?\.pkl)\"', r'"models/configs/\1"', content)

    # Replace yield_model.json -> models/configs/
    content = re.sub(r"'yield_model\.json'", r"'models/configs/yield_model.json'", content)
    content = re.sub(r'"yield_model\.json"', r'"models/configs/yield_model.json"', content)

    # Replace class_names, lstm_config, fusion_config
    for cfg in ['class_names.json', 'lstm_config.json', 'fusion_config.json']:
        if "models/configs/" not in content:
            content = re.sub(r"'" + cfg + r"'", r"'models/configs/" + cfg + r"'", content)
            content = re.sub(r'"' + cfg + r'"', r'"models/configs/' + cfg + r'"', content)

    # Replace .csv data -> data/
    content = re.sub(r'\'([^\']*?\.csv)\'', r"'data/\1'", content)
    content = re.sub(r'\"([^\']*?\.csv)\"', r'"data/\1"', content)

    file_path.write_text(content, encoding=encoding)
    print(f'Updated {filepath}')

process_file('dashboard/dashboard.py')
process_file('dashboard/dashboard_clean.py')
process_file('tests/tests.py')
process_file('tests/test_leaf_detect.py')
