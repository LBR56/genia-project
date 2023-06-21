from datetime import datetime
import os

def set_result_dir(result_dir, query = None, **kwargs):
    if query:
        result_dir = "src/"
        result_dir += datetime.now().strftime(r"%Y%m%d") + "/"
        result_dir += query + "/"

        if not os.path.exists("/temp/" + result_dir):
            os.makedirs("/temp/" + result_dir)
        
    elif not os.path.exists(result_dir):
        raise FileNotFoundError("파일이 존재하지 않습니다.")
    
    return result_dir