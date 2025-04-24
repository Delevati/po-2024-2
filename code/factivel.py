import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
import os
from matplotlib.patches import Patch

def load_data(file_path: str) -> dict:
    """Carrega os dados de parâmetros de otimização de um arquivo JSON."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo '{file_path}' não encontrado.")
    except json.JSONDecodeError:
        raise ValueError(f"Arquivo '{file_path}' contém JSON inválido.")

def create_constraints(mosaic_1: dict, mosaic_2: dict, relaxation: float) -> List[Callable]:
    """Cria as funções de restrição baseadas nos dados dos mosaicos e fator de relaxamento."""
    return [
        lambda x, y: x + y <= mosaic_1['estimated_coverage'] + relaxation,
        lambda x, y: x >= mosaic_1['quality_factor'] - relaxation,
        lambda x, y: y <= mosaic_2['estimated_coverage'] + relaxation,
        lambda x, y: y >= mosaic_2['quality_factor'] - relaxation
    ]

def find_feasible_region_and_optimal(constraints: List[Callable], 
                                     resolution: int = 100) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]], float]:
    """
    Encontra a região viável e a solução ótima para o problema de otimização.
    
    Args:
        constraints: Lista de funções que definem as restrições
        resolution: Resolução da malha para busca de soluções
        
    Returns:
        Tupla contendo (pontos viáveis, solução ótima, valor ótimo)
    """
    feasible_points = []
    optimal_solution = None
    max_value = -np.inf
    
    # Criamos uma matriz para visualizar valores da função objetivo
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    
    # Utilizamos uma única varredura para identificar pontos viáveis e solução ótima
    for x in x_vals:
        for y in y_vals:
            if all(constraint(x, y) for constraint in constraints):
                feasible_points.append((x, y))
                value = x + y  # Função objetivo (pode ser customizada)
                if value > max_value:
                    max_value = value
                    optimal_solution = (x, y)
    
    return feasible_points, optimal_solution, max_value

def visualize_solution(constraints: List[Callable], 
                       feasible_points: List[Tuple[float, float]], 
                       optimal_solution: Optional[Tuple[float, float]],
                       max_value: float,
                       relaxation: float,
                       mosaic_1: dict,
                       mosaic_2: dict,
                       resolution: int = 100,
                       save_path: Optional[str] = None) -> None:
    """Visualiza o espaço de soluções, restrições e a solução ótima."""
    # Configurar figura com tamanho adequado
    plt.figure(figsize=(10, 8))
    
    # Extrair limites das restrições para facilitar a visualização
    x_min = mosaic_1['quality_factor'] - relaxation
    y_min = mosaic_2['quality_factor'] - relaxation
    y_max = mosaic_2['estimated_coverage'] + relaxation
    sum_max = mosaic_1['estimated_coverage'] + relaxation
    
    # Imprimir diagnóstico das restrições
    print("\nDiagnóstico de Restrições:")
    print(f"Linha 1: x + y ≤ {sum_max:.2f} (linha diagonal)")
    print(f"Linha 2: x ≥ {x_min:.2f} (linha vertical)")
    print(f"Linha 3: y ≤ {y_max:.2f} (linha horizontal)")
    print(f"Linha 4: y ≥ {y_min:.2f} (linha horizontal)")
    
    # Calcular limites do gráfico - exatamente 0 a 1
    plot_x_max = 1.0
    plot_y_max = 1.0
    
    # Gerar malha para visualização
    x_vals = np.linspace(0, plot_x_max, resolution)
    y_vals = np.linspace(0, plot_y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Determinar a região viável
    feasible_region = np.ones_like(X, dtype=bool)
    for constraint in constraints:
        feasible_region &= constraint(X, Y)
    
    # Plotar região viável com sombreamento azul claro
    plt.contourf(X, Y, feasible_region, levels=[0, 1], colors=['#e6f2ff'], alpha=0.7)
    
    # Definir elemento para a legenda da região viável
    legend_elements = [Patch(facecolor='#e6f2ff', alpha=0.7, edgecolor='#0066cc', 
                       label='Feasible Region')]

    # Desenhar linhas de restrição com notação matemática adequada
    # Restrição 1: x + y = sum_max (diagonal)
    x_diag = np.array([0, sum_max])
    y_diag = np.array([sum_max, 0])
    plt.plot(x_diag, y_diag, '--', color='#000066', linewidth=1.0, 
            label=r'$x + y \leq %.2f$' % sum_max)

    # Restrição 2: x = x_min (vertical)
    plt.axvline(x=x_min, color='#660000', linestyle='--', linewidth=1.0,
                label=r'$x \geq %.2f$' % x_min)

    # Restrição 3: y = y_max (horizontal superior)
    plt.axhline(y=y_max, color='#006600', linestyle='--', linewidth=1.0,
                label=r'$y \leq %.2f$' % y_max)

    # Restrição 4: y = y_min (horizontal inferior)
    plt.axhline(y=y_min, color='#660066', linestyle='--', linewidth=1.0,
                label=r'$y \geq %.2f$' % y_min)

    # Anotações posicionadas estrategicamente para evitar sobreposições
    plt.annotate(r'$x + y = %.2f$' % sum_max, xy=(sum_max/2, sum_max/2), 
                xytext=(10, 10), textcoords='offset points', color='#000066',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
                
    plt.annotate(r'$x = %.2f$' % x_min, xy=(x_min, plot_y_max/2), 
                xytext=(5, 0), textcoords='offset points', color='#660000',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
                
    plt.annotate(r'$y = %.2f$' % y_max, xy=(plot_x_max/2, y_max), 
                xytext=(0, 5), textcoords='offset points', color='#006600',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
                
    plt.annotate(r'$y = %.2f$' % y_min, xy=(plot_x_max/2, y_min), 
                xytext=(0, -15), textcoords='offset points', color='#660066',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # Plotar os pontos binários com estilo mais científico
    binary_points = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    plt.scatter(binary_points[:, 0], binary_points[:, 1], color='#cc0000', 
                s=40, marker='o', edgecolors='black', linewidth=0.5, 
                label='Binary Points')  
    
    # Adicionar labels aos pontos binários
    for point in binary_points:
        plt.annotate(f"({point[0]}, {point[1]})", 
                     xy=point, 
                     xytext=(5, 5), 
                     textcoords='offset points')
    
    # Destacar a solução ótima
    if optimal_solution:
        plt.scatter(*optimal_solution, color='green', s=100, 
                    label=f'Solução Ótima: ({optimal_solution[0]:.2f}, {optimal_solution[1]:.2f})')
        plt.annotate(f"Valor: {max_value:.2f}", 
                     xy=optimal_solution, 
                     xytext=(10, -10), 
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.8))
    
    # Adicionar anotações e formatação
    plt.title('Feasible Region and Optimal Solution Analysis', fontsize=12)
    plt.xlabel('Variable $x$ (Mosaic 1)', fontsize=10)
    plt.ylabel('Variable $y$ (Mosaic 2)', fontsize=10)
    
    # Definir limites exatos de 0 a 1
    plt.xlim(0.0, plot_x_max)
    plt.ylim(0.0, plot_y_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Criar legenda personalizada incluindo o elemento da região viável
    handles, labels = plt.gca().get_legend_handles_labels()
    all_handles = legend_elements + handles
    all_labels = [h.get_label() for h in all_handles]
    by_label = dict(zip(all_labels, all_handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)
    
    # Salvar o gráfico se um caminho for fornecido
    if save_path:
        plt.grid(True, linestyle=':', alpha=0.4, color='gray')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Gráfico salvo em {os.path.abspath(save_path)}")
    
    # Mostrar o gráfico
    plt.tight_layout()
    plt.show()

def create_relaxed_constraints(mosaic_1: dict, mosaic_2: dict, relaxations: dict) -> List[Callable]:
    """Cria restrições com relaxamentos customizados para cada tipo de restrição."""
    return [
        lambda x, y: x + y <= mosaic_1['estimated_coverage'] + relaxations.get('sum', 0.3),
        lambda x, y: x >= mosaic_1['quality_factor'] - relaxations.get('quality_1', 0.4),
        lambda x, y: y <= mosaic_2['estimated_coverage'] + relaxations.get('coverage_2', 0.3),
        lambda x, y: y >= mosaic_2['quality_factor'] - relaxations.get('quality_2', 0.4)
    ]

def main() -> None:
    """Função principal do programa de otimização."""
    # Parâmetros configuráveis
    file_path = 'optimization_parameters.json'
    
    # Você pode escolher entre relaxamento uniforme ou personalizado
    use_custom_relaxation = False
    
    # Relaxamento uniforme (usado se use_custom_relaxation = False)
    relaxation = 0.33
    
    # Relaxamentos personalizados (usados se use_custom_relaxation = True)
    relaxations = {
        'sum': 0.3,        # Para restrição x + y <= ...
        'quality_1': 0.5,  # Para restrição x >= ... (maior relaxamento)
        'coverage_2': 0.3, # Para restrição y <= ...
        'quality_2': 0.5   # Para restrição y >= ... (maior relaxamento)
    }
    
    resolution = 100
    save_fig = True
    output_path = 'optimization_visualization.png'
    
    try:
        # Carregar dados
        data = load_data(file_path)
        
        # Extrair informações dos grupos de mosaico
        mosaic_groups = data['mosaic_groups']
        mosaic_1 = next((m for m in mosaic_groups if m['group_id'] == 'mosaic_1'), None)
        mosaic_2 = next((m for m in mosaic_groups if m['group_id'] == 'mosaic_2'), None)
        
        if not mosaic_1 or not mosaic_2:
            raise ValueError("Os grupos de mosaico 1 ou 2 não foram encontrados no JSON.")
        
        # Imprimir valores originais dos parâmetros
        print("\nValores originais dos parâmetros:")
        print(f"mosaic_1['estimated_coverage'] = {mosaic_1['estimated_coverage']}")
        print(f"mosaic_1['quality_factor'] = {mosaic_1['quality_factor']}")
        print(f"mosaic_2['estimated_coverage'] = {mosaic_2['estimated_coverage']}")
        print(f"mosaic_2['quality_factor'] = {mosaic_2['quality_factor']}")
        
        # Criar restrições (com relaxamento uniforme ou personalizado)
        if use_custom_relaxation:
            constraints = create_relaxed_constraints(mosaic_1, mosaic_2, relaxations)
            effective_relaxation = relaxations  # Para visualização
            print("\nRestrições com relaxamentos personalizados:")
            print(f"x + y <= {mosaic_1['estimated_coverage']} + {relaxations['sum']}")
            print(f"x >= {mosaic_1['quality_factor']} - {relaxations['quality_1']}")
            print(f"y <= {mosaic_2['estimated_coverage']} + {relaxations['coverage_2']}")
            print(f"y >= {mosaic_2['quality_factor']} - {relaxations['quality_2']}")
        else:
            constraints = create_constraints(mosaic_1, mosaic_2, relaxation)
            effective_relaxation = relaxation  # Para visualização
            print("\nRestrições com relaxamento uniforme:")
            print(f"x + y <= {mosaic_1['estimated_coverage']} + {relaxation}")
            print(f"x >= {mosaic_1['quality_factor']} - {relaxation}")
            print(f"y <= {mosaic_2['estimated_coverage']} + {relaxation}")
            print(f"y >= {mosaic_2['quality_factor']} - {relaxation}")
        
        # Encontrar região viável e solução ótima
        feasible_points, optimal_solution, max_value = find_feasible_region_and_optimal(
            constraints, resolution)
        
        print(f"\nNúmero de pontos viáveis: {len(feasible_points)}")
        
        # Exibir resultados da otimização
        if optimal_solution:
            plt.scatter(*optimal_solution, color='#008800', s=80, marker='*', 
                        edgecolors='black', linewidth=0.5,
                        label=r'Optimal Solution: $(%.2f, %.2f)$' % optimal_solution)
            plt.annotate(r'Value: $%.2f$' % max_value, 
                        xy=optimal_solution, 
                        xytext=(10, -10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#008800", alpha=0.9),
                 fontsize=9)
        
        # Visualizar resultados
        visualize_solution(
            constraints, 
            feasible_points, 
            optimal_solution, 
            max_value,
            effective_relaxation,
            mosaic_1,
            mosaic_2,
            resolution,
            output_path if save_fig else None
        )
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        raise

if __name__ == "__main__":
    main()