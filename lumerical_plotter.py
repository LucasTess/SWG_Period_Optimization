import numpy as np
import matplotlib.pyplot as plt
import os  # Biblioteca adicionada para manipular pastas e caminhos

# Configurações do arquivo e do espectro
file_path = 'optimization-result-export.txt'  # Altere para o nome exato do seu arquivo txt
lambda_start = 1400  # Comprimento de onda inicial em nm
lambda_end = 1600    # Comprimento de onda final em nm

# Lista para armazenar os valores de S11
s11_values = []

# Leitura do arquivo de texto
try:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Ignora a primeira linha (cabeçalho) e converte o restante para float
        for line in lines[1:]:
            cleaned_line = line.strip()
            if cleaned_line:  # Verifica se a linha não está vazia
                s11_values.append(float(cleaned_line))
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()

# Geração do vetor de comprimentos de onda
num_points = len(s11_values)
wavelengths = np.linspace(lambda_start, lambda_end, num_points)

# Criação da figura e do gráfico
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, s11_values, color='#3498db', linewidth=1.5, label='Filtro Projetado')

# Estilização do gráfico
#plt.title('Espectro de Reflexão/Transmissão ($S_{11}$)')
plt.xlabel('Wavelength(nm)', fontsize=12, weight='bold')
plt.ylabel(r'${S_{11}}^2$', fontsize=12, weight='bold')


# Configuração da grade (grid)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

# Limites dos eixos
plt.xlim(lambda_start, lambda_end)
plt.ylim(0, 1.05)

# Otimização do layout
plt.tight_layout()

# --- Configuração de exportação para a pasta específica ---
output_dir = 'compilation exports'

# Cria a pasta se ela não existir no diretório atual
os.makedirs(output_dir, exist_ok=True)

# Define o caminho completo unindo a pasta e o nome do arquivo
output_filename = os.path.join(output_dir, 'grafico_S11.pdf')

# Exportar para PDF
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print(f"Gráfico gerado e salvo com sucesso em '{output_filename}'!")

# Mostrar o gráfico na tela (opcional)
plt.show()