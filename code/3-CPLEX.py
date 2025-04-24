import os
import json
from docplex.mp.model import Model
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import shutil
from pathlib import Path

# Diretórios principais
METADATA_DIR = "/Volumes/luryand/coverage_otimization/metadata"
OUTPUT_DIR = "/Volumes/luryand/coverage_otimization/results"
SELECTED_MOSAICS_DIR = os.path.join(OUTPUT_DIR, "selected_mosaics")
OPTIMIZATION_PARAMS_FILE = os.path.join(METADATA_DIR, 'optimization_parameters.json')
CPLEX_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'cplex_selected_mosaic_groups.json')
OUTPUT_BASE_DIR = Path("/Volumes/luryand/coverage_otimization")  # Base do projeto

# Criar diretórios necessários
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SELECTED_MOSAICS_DIR, exist_ok=True)

def solve_mosaic_selection_milp(optimization_params):
    """
    Implementa e resolve o modelo MILP para seleção de grupos de mosaicos,
    baseado nos dados de 'optimization_parameters.json' e na formulação do artigo.
    """
    if not optimization_params.get('mosaic_groups'):
        print("Erro: Nenhum 'mosaic_groups' encontrado nos parâmetros de otimização.")
        return []

    mosaic_groups = optimization_params['mosaic_groups']
    image_catalog = optimization_params.get('image_catalog', []) # Pode ser útil para restrições

    # --- Criação do Modelo ---
    mdl = Model(name='selecao_mosaicos_apa')

    # --- Variáveis de Decisão ---
    # y_g = 1 se o grupo de mosaico g for selecionado, 0 caso contrário
    y = {}
    for group in mosaic_groups:
        group_id = group['group_id']
        # Usar group_id como chave e nome da variável
        y[group_id] = mdl.binary_var(name=f'y_{group_id}')
    print(f"Número de variáveis de decisão (grupos de mosaico): {len(y)}")
    if not y:
        print("Nenhuma variável de decisão criada. Verifique a lista 'mosaic_groups'.")
        return []

    # --- Parâmetros (da Função Objetivo do Artigo) ---
    alpha = 0.05  # Penalidade por número de grupos (incentiva menos grupos)
    beta = 0.1   # Penalidade por usar SAR (AJUSTE CONFORME NECESSÁRIO - maior que alpha?)

    # --- Função Objetivo (Baseada no Artigo, com beta) ---
    # Maximizar: sum(E_g * Q_g * y_g) - alpha * sum(y_g) - beta * sum(y_g se grupo g contém SAR)
    # Onde E_g (Efetividade/Cobertura) é 'estimated_coverage'
    # E Q_g (Qualidade) é 'quality_factor'
    total_coverage_quality = mdl.sum(
        group['estimated_coverage'] * group['quality_factor'] * y[group['group_id']]
        for group in mosaic_groups if group['group_id'] in y # Garante que o grupo existe nas variáveis
    )

    # Penalidade pelo número de grupos selecionados
    penalty_num_groups = alpha * mdl.sum(y[group_id] for group_id in y)

    # Penalidade pelo uso de SAR (lendo o campo 'contains_sar' do JSON)
    # Assume que 'contains_sar': True foi adicionado no test-version.py para todos os grupos
    penalty_sar_usage = beta * mdl.sum(
        y[group['group_id']]
        for group in mosaic_groups
        if group['group_id'] in y and group.get('contains_sar', False) # Verifica o campo adicionado
    )

    mdl.maximize(total_coverage_quality - penalty_num_groups - penalty_sar_usage) # Subtrai ambas penalidades
    print(f"Função objetivo definida: Maximizar (Cobertura * Qualidade) - alpha({alpha}) * (Num Grupos) - beta({beta}) * (Uso SAR)") # Atualiza print

    # --- Restrições (Baseadas no Artigo e nos Dados) ---
    print("Adicionando restrições...")

    # 1. Restrição de Capacidade - Removida a restrição arbitrária de 10 grupos
    # Permitimos que o modelo selecione o número ótimo de grupos
    max_selected_groups = len(y)  # Permitir até o número total de grupos disponíveis
    print(f"  - Sem restrição de número máximo de grupos. Limite natural: {max_selected_groups} grupos.")

    # 2. Restrição de Conflito de Imagem:
    #    Se uma imagem pertence a múltiplos grupos, selecionar no máximo um desses grupos.
    #    Isso evita selecionar mosaicos muito redundantes baseados na mesma imagem chave.
    image_to_groups = defaultdict(list)
    for group in mosaic_groups:
        group_id = group['group_id']
        if group_id not in y: continue # Ignora grupos sem variável
        for image_filename in group.get('images', []):
            image_to_groups[image_filename].append(group_id)

    conflict_constraints_added = 0
    for image_filename, groups_with_image in image_to_groups.items():
        if len(groups_with_image) > 1:
            # Se uma imagem está em mais de um grupo, a soma das variáveis y desses grupos deve ser <= 1
            mdl.add_constraint(mdl.sum(y[group_id] for group_id in groups_with_image) <= 1,
                               ctname=f"conflict_{image_filename.replace('.','_').replace('-','_')[:30]}") # Nome curto e válido
            conflict_constraints_added += 1
    if conflict_constraints_added > 0:
        print(f"  - Restrições adicionadas: {conflict_constraints_added} restrições de conflito de imagem (máx 1 grupo por imagem compartilhada).")
    else:
        print("  - Nenhuma restrição de conflito de imagem necessária (nenhuma imagem compartilhada entre grupos).")


    # 3. Restrição de Cobertura Mínima Total:
    #    Garante que a soma das coberturas estimadas dos grupos selecionados atinja um valor mínimo.
    min_total_estimated_coverage = 0.90 # Exigir que a soma das coberturas estimadas seja pelo menos 90%
    if min_total_estimated_coverage > 0:
        mdl.add_constraint(
            mdl.sum(group['estimated_coverage'] * y[group['group_id']]
                    for group in mosaic_groups if group['group_id'] in y) >= min_total_estimated_coverage,
            ctname="min_total_coverage"
        )
        print(f"  - Restrição adicionada: Cobertura total estimada >= {min_total_estimated_coverage:.1%}")
    else:
         print("  - Aviso: Restrição de cobertura mínima total não aplicada (valor <= 0).")


    # --- REMOVIDO/COMENTADO: Restrição de Cobertura por APA Individual ---
    print("  - Restrição de cobertura por APA individual NÃO está sendo aplicada.")

    # --- Resolver o Modelo ---
    print("\nIniciando a resolução do modelo CPLEX...")
    solution = mdl.solve()

    # --- Analisar e Retornar os Resultados ---
    selected_groups_details = []
    if solution:
        print("\n--- Solução Encontrada ---")
        objective_value = solution.get_objective_value()
        print(f"Valor da Função Objetivo: {objective_value:.4f}")
        selected_count = 0
        total_selected_coverage = 0
        selected_group_ids = []

        for group in mosaic_groups:
            group_id = group['group_id']
            if group_id in y and solution.get_value(y[group_id]) > 0.9: # Usar > 0.9 para segurança com binárias
                selected_count += 1
                selected_group_ids.append(group_id)
                selected_groups_details.append(group) # Adiciona o dicionário completo do grupo selecionado
                total_selected_coverage += group.get('estimated_coverage', 0)

        print(f"\nTotal de grupos de mosaico selecionados: {selected_count}")
        if selected_count > 0:
            print(f"IDs dos Grupos Selecionados: {', '.join(selected_group_ids)}")
            print(f"Soma da Cobertura Estimada dos grupos selecionados: {total_selected_coverage:.2%}")
        else:
             print("Aviso: Nenhuma solução viável encontrada que selecione algum grupo (objetivo pode ser 0 ou negativo, ou restrições muito fortes).")

    else:
        print("\n--- Nenhuma Solução Encontrada ---")
        solve_status = mdl.get_solve_status()
        if solve_status:
            print(f"Status da Solução CPLEX: {solve_status}")
            if "Infeasible" in str(solve_status):
                print("  -> O modelo é infactível. Verifique se as restrições são contraditórias.")
            elif "Unbounded" in str(solve_status):
                 print("  -> O modelo é ilimitado. Verifique a função objetivo e restrições.")
        else:
            print("Status da solução não disponível.")
        print("Verifique as restrições, a função objetivo e os dados de entrada.")

    return selected_groups_details

def plot_selected_groups(selected_groups, output_dir):
    """
    Gera um gráfico da solução encontrada, mostrando os grupos selecionados pelo CPLEX.
    """
    if not selected_groups:
        print("Nenhum grupo selecionado para gerar o gráfico.")
        return
    
    # Preparar dados para o gráfico
    group_ids = [group['group_id'] for group in selected_groups]
    coverages = [group['estimated_coverage'] for group in selected_groups]
    qualities = [group['quality_factor'] for group in selected_groups]
    
    # Ordenar os grupos por cobertura para melhor visualização
    idx_sorted = np.argsort(coverages)[::-1]
    group_ids = [group_ids[i] for i in idx_sorted]
    qualities = [qualities[i] for i in idx_sorted]
    coverages = [coverages[i] for i in idx_sorted]
    
    # Calcular valor combinado para representar o "valor" de cada grupo na solução
    combined_values = [cov * qual for cov, qual in zip(coverages, qualities)]
    
    # Criar figura e eixos
    plt.figure(figsize=(12, 8), dpi=150)
    
    # Plotar as linhas de cobertura e qualidade
    x = range(len(group_ids))
    plt.plot(x, coverages, 'bo-', linewidth=2, markersize=8, label='Cobertura Estimada')
    plt.plot(x, qualities, 'rs-', linewidth=2, markersize=8, label='Fator de Qualidade')
    plt.plot(x, combined_values, 'gd--', linewidth=1.5, markersize=7, label='Valor Combinado')
    
    # Configurações visuais
    plt.title('Grupos de Mosaico Selecionados pelo CPLEX', fontsize=16, fontweight='bold')
    plt.xlabel('Grupos Selecionados (Ordenados por Cobertura)', fontsize=14)
    plt.ylabel('Valor (0-1)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0, max(max(coverages), max(qualities), max(combined_values)) * 1.1])
    plt.xticks(x, group_ids, rotation=45, ha='right')
    
    # Adicionar valores acima dos pontos de cobertura
    for i, cov in enumerate(coverages):
        plt.text(x[i], cov + 0.02, f'{cov:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Adicionar informação total
    plt.figtext(0.5, 0.01, 
                f"Cobertura Total: {sum(coverages):.2f} | Qualidade Média: {sum(qualities)/len(qualities):.2f} | Grupos: {len(group_ids)}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.legend(loc='best')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Ajustar para acomodar o texto no rodapé
    
    # Salvar gráfico
    plot_path = os.path.join(output_dir, 'cplex_solution_graph.png')
    plt.savefig(plot_path)
    print(f"Gráfico da solução salvo em: {plot_path}")
    plt.close()
    
    # Segunda visualização: Mapa de calor mostrando relação cobertura-qualidade
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Criar pontos com tamanhos representando o número de imagens em cada grupo
    image_counts = [len(group.get('images', [])) for group in selected_groups]
    sizes = [count * 30 for count in image_counts]  # Ajustar escala conforme necessário
    
    # Diagrama de dispersão colorido pelo valor combinado
    scatter = plt.scatter(coverages, qualities, s=sizes, c=combined_values, 
                         cmap='viridis', alpha=0.8, edgecolors='black')
    
    # Conectar pontos com uma linha pontilhada para visualizar a sequência
    plt.plot(coverages, qualities, 'k--', alpha=0.5, linewidth=1)
    
    # Adicionar rótulos aos pontos
    for i, group_id in enumerate(group_ids):
        plt.annotate(group_id, (coverages[i], qualities[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Configurações visuais
    plt.colorbar(scatter, label='Valor Combinado (Cobertura × Qualidade)')
    plt.title('Relação Cobertura-Qualidade dos Grupos Selecionados', fontsize=16, fontweight='bold')
    plt.xlabel('Cobertura Estimada', fontsize=14)
    plt.ylabel('Fator de Qualidade', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Legenda para o tamanho dos pontos
    sizes_legend = [min(image_counts), max(image_counts)]
    labels = [f'{size} imagens' for size in sizes_legend]
    
    # Criar elementos de legenda personalizados
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=np.sqrt(count*30)/5, label=label)  # Ajustar divisor para tamanho visual adequado
        for count, label in zip(sizes_legend, labels)
    ]
    
    plt.legend(handles=legend_elements, title="Tamanho = Nº de Imagens", loc='best')
    plt.tight_layout()
    
    # Salvar o segundo gráfico
    scatter_path = os.path.join(output_dir, 'cplex_coverage_quality_scatter.png')
    plt.savefig(scatter_path)
    print(f"Gráfico de dispersão cobertura-qualidade salvo em: {scatter_path}")
    plt.close()

def find_image_metadata(image_filename, all_metadata_file):
    """
    Busca metadados de uma imagem específica no arquivo de metadados completo.
    Retorna None se não encontrar.
    """
    try:
        # Carrega todos os metadados das imagens processadas
        with open(all_metadata_file, 'r') as f:
            all_images_metadata = json.load(f)
        
        # Busca a imagem específica
        for img_meta in all_images_metadata:
            if img_meta.get('filename') == image_filename:
                return img_meta
                
        print(f"  Aviso: Metadados para imagem '{image_filename}' não encontrados")
        return None
    except Exception as e:
        print(f"  Erro ao buscar metadados para '{image_filename}': {e}")
        return None

def copy_mosaic_files_to_results(selected_groups, output_dir, all_metadata_file):
    """
    Copia os arquivos TCI das imagens dos grupos selecionados para a pasta de resultados.
    """
    print(f"\nCopiando visualizações de mosaicos...")

    if not selected_groups:
        print("  Nenhum grupo selecionado para copiar visualizações.")
        return False

    plots_source_dir = OUTPUT_BASE_DIR / 'results' / 'plots'
    plots_all_dir = Path(OUTPUT_DIR) / 'plots'

    if not plots_source_dir.exists():
        print(f"  Aviso: Diretório de plots '{plots_source_dir}' não encontrado. Nenhuma visualização será copiada.")
        return False
    
    num_groups = len(selected_groups)
    copied_images = 0
    skipped_images = 0
    
    # Processa cada grupo selecionado
    for i, group in enumerate(selected_groups):
        group_id = group.get('group_id', f"grupo_{i+1}")
        group_dir = os.path.join(output_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)
                
        image_filenames = group.get('images', [])
        if not image_filenames:
            print(f"    Aviso: Nenhuma imagem encontrada no grupo {group_id}")
            continue
            
        for image_filename in image_filenames:
            img_meta = find_image_metadata(image_filename, all_metadata_file)
            if not img_meta:
                skipped_images += 1
                continue
                
            tci_path = img_meta.get('tci_path') or img_meta.get('temp_tci_path')
            if tci_path and os.path.exists(tci_path):
                tci_filename = os.path.basename(tci_path)
                dest_tci = os.path.join(group_dir, tci_filename)
                try:
                    shutil.copy2(tci_path, dest_tci)
                    copied_images += 1
                except Exception as e:
                    print(f"    Erro ao copiar {tci_path} para {dest_tci}: {e}")
                    skipped_images += 1
            else:
                print(f"    Aviso: TCI não encontrado para {image_filename}")
                skipped_images += 1
                
        # Salva um arquivo JSON com as informações do grupo
        group_info_file = os.path.join(group_dir, f"{group_id}_info.json") 
        try:
            with open(group_info_file, 'w') as f:
                json.dump(group, f, indent=2)
        except Exception as e:
            print(f"    Erro ao salvar informações do grupo {group_id}: {e}")
    
    print(f"\nResultado da cópia: {copied_images} imagens copiadas, {skipped_images} imagens ignoradas")
    return copied_images > 0

def copy_mosaic_visual_plots(selected_groups, output_dir):
    """
    Copia os plots/visualizações de mosaicos previamente gerados pelo script 2
    para a pasta de resultados correspondente a cada grupo selecionado E para
    uma pasta central 'plots' dentro de OUTPUT_DIR.
    Usa o padrão de nome de arquivo exato do script 2 para a busca.
    """
    print(f"\nCopiando visualizações de mosaicos (usando padrão de nome com data)...")

    if not selected_groups:
        print("  Nenhum grupo selecionado para copiar visualizações.")
        return False

    plots_source_dir = OUTPUT_BASE_DIR / 'publication_plots'
    plots_all_dir = Path(OUTPUT_DIR) / 'plots'

    if not plots_source_dir.exists():
        print(f"  Aviso: Diretório de plots '{plots_source_dir}' não encontrado. Nenhuma visualização será copiada.")
        return False

    plots_all_dir.mkdir(exist_ok=True)
    print(f"  Procurando plots em: {plots_source_dir}")
    print(f"  Diretório central para plots copiados: {plots_all_dir}")

    copied_plots_count = 0
    found_plot_for_group = False

    for group in selected_groups:
        group_id = group.get('group_id', '')
        group_dest_dir = Path(output_dir) / group_id # Usar Path para consistência
        group_dest_dir.mkdir(exist_ok=True)
        found_plot_for_group = False # Resetar flag para cada grupo

        # Extrair data do grupo (usar time_window_start)
        date_str = "NODATE"
        try:
            start_date_iso = group.get('time_window_start')
            if start_date_iso:
                date_str = start_date_iso.split('T')[0] # Pega YYYY-MM-DD
            else:
                 print(f"  Aviso: Grupo {group_id} não tem 'time_window_start'. Não é possível usar data na busca.")
                 continue # Pula para o próximo grupo se não tiver data
        except Exception as e:
            print(f"  Erro ao extrair data para {group_id}: {e}. Pulando busca para este grupo.")
            continue # Pula para o próximo grupo

        # Determinar o padrão do arquivo de plot com base no ID do grupo e data
        try:
            if group_id.startswith('mosaic_'):
                mosaic_num_str = group_id.split('_')[-1]
                if mosaic_num_str.isdigit():
                    mosaic_num = int(mosaic_num_str)

                    plot_pattern = f"{date_str}_good_mosaic_{mosaic_num}_*_raster.png"
                    plot_files_found = list(plots_source_dir.glob(plot_pattern))

                    if plot_files_found:
                        source_plot_path = plot_files_found[0]
                        dest_filename_with_date = f"{date_str}_{group_id}_visual.png"
                        dest_plot_path_group = group_dest_dir / dest_filename_with_date
                        dest_plot_path_all = plots_all_dir / dest_filename_with_date

                        try:
                            shutil.copy2(source_plot_path, dest_plot_path_group)

                            shutil.copy2(source_plot_path, dest_plot_path_all)

                            copied_plots_count += 1 # Contar apenas uma vez por grupo encontrado
                            found_plot_for_group = True # Marcar que encontramos um plot
                        except Exception as e:
                            print(f"  Erro ao copiar visualização {source_plot_path.name} para {group_id} ou central: {e}")
                    else:
                         pass
                else:
                    print(f"  Aviso: Não foi possível extrair número do ID '{group_id}' para buscar visualização.")
            else:
                print(f"  Aviso: ID de grupo '{group_id}' não segue o padrão esperado 'mosaic_X'. Não é possível buscar visualização.")

        except Exception as e:
            print(f"  Erro inesperado ao processar visualizações para grupo {group_id}: {e}")

        # Log se não encontrou plot para o grupo
        if not found_plot_for_group:
             print(f"  Nenhuma visualização correspondente encontrada para o grupo {group_id} com o padrão esperado em {plots_source_dir}.")

    if copied_plots_count > 0:
        print(f"\nTotal de visualizações de mosaicos copiadas (para pastas de grupo e central): {copied_plots_count}")
    else:
        print("\nNenhuma visualização de mosaico encontrada ou copiada.")
    return copied_plots_count > 0

def save_cplex_results(selected_groups, output_filepath):
    """
    Salva os detalhes dos grupos de mosaicos selecionados pelo CPLEX em um arquivo JSON,
    gera gráficos e copia os arquivos TCI e visualizações para a pasta de resultados.
    """
    output_dir = os.path.dirname(output_filepath)
    all_metadata_file = os.path.join(METADATA_DIR, 'all_processed_images_log.json')

    # Salva os resultados em JSON
    try:
        with open(output_filepath, 'w') as f:
            json.dump(selected_groups, f, indent=4)
        print(f"\nResultados da otimização CPLEX salvos em: {output_filepath}")
    except Exception as e:
        print(f"Erro ao salvar os resultados do CPLEX em {output_filepath}: {e}")
        return False

    # Gera os gráficos de visualização da solução CPLEX (não os plots de mosaico)
    try:
        plot_selected_groups(selected_groups, output_dir)
    except Exception as e:
        print(f"Erro ao gerar gráficos da solução CPLEX: {e}")

    # Copia os arquivos TCI para o diretório de resultados
    try:
        copy_mosaic_files_to_results(selected_groups, SELECTED_MOSAICS_DIR, all_metadata_file)
    except Exception as e:
        print(f"Erro ao copiar arquivos TCI dos mosaicos selecionados: {e}")
        # Continuar mesmo se a cópia de TCI falhar, para tentar copiar plots

    # Copia as visualizações de mosaicos do script 2 (com data e para pasta central)
    try:
        copy_mosaic_visual_plots(selected_groups, SELECTED_MOSAICS_DIR)
    except Exception as e:
        print(f"Erro ao copiar visualizações de mosaicos: {e}")

    return True

def main():
    """
    Função principal para carregar dados, executar o modelo MILP e salvar resultados.
    """
    print("--- Iniciando Script de Otimização CPLEX (Seleção de Mosaicos) ---")

    # Carregar os parâmetros de otimização do arquivo JSON gerado por test-version.py
    try:
        print(f"Carregando parâmetros de: {OPTIMIZATION_PARAMS_FILE}")
        with open(OPTIMIZATION_PARAMS_FILE, 'r') as f:
            optimization_params = json.load(f)
        print("Parâmetros carregados com sucesso.")
        # Validação básica da estrutura
        if 'mosaic_groups' not in optimization_params:
             print("AVISO: Chave 'mosaic_groups' não encontrada no JSON. Otimização pode falhar.")
        if 'image_catalog' not in optimization_params:
             print("AVISO: Chave 'image_catalog' não encontrada no JSON.")

    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo de parâmetros '{OPTIMIZATION_PARAMS_FILE}' não encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Erro Crítico: Falha ao decodificar o JSON do arquivo '{OPTIMIZATION_PARAMS_FILE}'. Verifique o formato.")
        return
    except Exception as e:
        print(f"Erro Crítico inesperado ao carregar parâmetros: {e}")
        return

    # Resolver o modelo MILP
    selected_mosaic_groups = solve_mosaic_selection_milp(optimization_params)

    # Salvar os resultados (lista de dicionários dos grupos selecionados) e gerar gráficos
    if selected_mosaic_groups:
        save_cplex_results(selected_mosaic_groups, CPLEX_RESULTS_FILE)
    else:
        print("\nNenhum grupo de mosaico foi selecionado pela otimização ou ocorreu um erro durante a resolução.")

    print("\n--- Script de Otimização CPLEX Finalizado ---")
    print(f"Os mosaicos selecionados foram copiados para: {SELECTED_MOSAICS_DIR}")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()