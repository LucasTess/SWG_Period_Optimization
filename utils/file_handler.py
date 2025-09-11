# utils/file_handler.py (Modificado para deletar apenas o conteúdo das subpastas)

import os
import shutil
import time

def delete_directory_contents(directory_path, retries=5, delay=1):
    """
    Limpa o conteúdo de um diretório.
    - Se encontrar arquivos no diretório principal, eles são deletados.
    - Se encontrar subdiretórios, entra neles e deleta APENAS OS ARQUIVOS,
      preservando a estrutura de pastas.
    Inclui lógica de múltiplas tentativas para lidar com arquivos bloqueados.

    Args:
        directory_path (str): O caminho do diretório a ser esvaziado.
        retries (int): O número máximo de tentativas de exclusão.
        delay (int): O tempo de espera em segundos entre as tentativas.
    """
    if not os.path.exists(directory_path):
        print(f"Aviso: Diretório não encontrado, não é possível limpar: {directory_path}")
        return

    print(f"Esvaziando conteúdo de: {directory_path} (preservando subpastas)...")
    
    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)

        try:
            # Caso 1: O item é um arquivo no diretório principal (ex: .fsp, .log)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                _delete_file_with_retry(item_path, item_name, retries, delay)

            # Caso 2: O item é um diretório (ex: ..._s-parametersweep)
            elif os.path.isdir(item_path):
                print(f"  - Limpando arquivos dentro do subdiretório: {item_name}")
                # os.walk percorre recursivamente todos os arquivos em todas as subpastas
                for root, dirs, files in os.walk(item_path):
                    for filename in files:
                        file_to_delete = os.path.join(root, filename)
                        _delete_file_with_retry(file_to_delete, filename, retries, delay)
                        
        except Exception as e:
            print(f"  - ERRO INESPERADO ao processar o item '{item_path}'. Motivo: {e}")


def _delete_file_with_retry(file_path, filename, retries, delay):
    """
    Função auxiliar que tenta deletar um único arquivo com múltiplas tentativas.
    """
    for attempt in range(retries):
        try:
            os.unlink(file_path)
            # Se a exclusão funcionou, sai do loop
            return
        except PermissionError:
            if attempt < retries - 1:
                print(f"    - Acesso negado a '{filename}'. Tentando novamente em {delay}s...")
                time.sleep(delay)
            else:
                print(f"    - ERRO FINAL: Não foi possível deletar o arquivo {filename} após {retries} tentativas.")
        except FileNotFoundError:
            # O arquivo já foi deletado por outro processo, o que é ok.
            break
        except Exception as e:
            print(f"    - ERRO INESPERADO ao deletar o arquivo {filename}: {e}")
            break

