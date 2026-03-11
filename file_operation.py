import json
import os


# def get_file_sha256(file_path, chunk_size=8192):
#     import hashlib

#     if not os.path.isfile(file_path):
#         return None

#     sha256 = hashlib.sha256()
#     with open(file_path, "rb") as f:
#         while chunk := f.read(chunk_size):
#             sha256.update(chunk)
#     return sha256.hexdigest()


def load_from_json(path: str) -> dict | None:
    """
    从 JSON 文件加载数据。

    Parameters
    ----------
    path : str
        JSON 文件路径

    Returns
    -------
    dict | None
        如果文件存在返回 JSON 数据，否则返回 None
    """

    # 判断文件是否存在
    if os.path.isfile(path):
        print("[INFO] load from file...")

        # 读取 JSON 文件
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def save_to_json(path: str, data):
    """
    保存数据到 JSON 文件。

    Parameters
    ----------
    path : str
        输出 JSON 文件路径
    data :
        要保存的数据
    """

    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 写入 JSON 文件
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
