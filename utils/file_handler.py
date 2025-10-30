# file_handler.py
import os

def clean_simulation_directory(directory_path, file_extension=None):
    """
    Limpa todos os arquivos em um diretório com uma extensão específica.
    
    Args:
        directory_path (str): O caminho do diretório a ser limpo.
        file_extension (str): A extensão dos arquivos a serem removidos (ex: '.h5', '.fsp').
                               Se for None, todos os arquivos serão removidos.
    """
    if not os.path.exists(directory_path):
        return
        
    print(f"Limpando o diretório de simulação: {directory_path} (arquivos com extensão: {file_extension})...")
    for filename in os.listdir(directory_path):
        if file_extension and not filename.endswith(file_extension):
            continue
            
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao remover o arquivo {file_path}: {e}")