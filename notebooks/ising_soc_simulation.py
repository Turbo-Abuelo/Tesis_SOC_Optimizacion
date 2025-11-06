import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ==========================================================
#                      PARÁMETROS AJUSTABLES
# ==========================================================
L = 16              # Tamaño de la retícula (e.g., 16x16, 32x32).
N_SPINS = L * L     
N_TRIALS = 3        # Número de repeticiones para promediar (n)
MAX_FLIPS_PER_SPIN = 10**3 # Flips por Spin (F/S). Para pruebas rápidas: 10^3.
MAX_STEPS = N_SPINS * MAX_FLIPS_PER_SPIN 

print(f"Iniciando simulación Ising ({L}x{L}). Pasos totales por ensayo: {MAX_STEPS:.1e}")
# ==========================================================
#                   FUNCIONES DEL MODELO ISING
# ==========================================================

def initialize_ising_model(L):
    """Inicializa espines y acoplamientos J (Spin Glass)."""
    spins = np.random.choice([-1, 1], size=(L, L))
    Jx = np.random.choice([-1, 1], size=(L, L))
    Jy = np.random.choice([-1, 1], size=(L, L))
    return spins, Jx, Jy

def calculate_energy(spins, Jx, Jy):
    """Calcula la energía total E (Condiciones Periódicas)."""
    E = 0
    E -= np.sum(Jx * spins * np.roll(spins, -1, axis=1))
    E -= np.sum(Jy * spins * np.roll(spins, -1, axis=0))
    return E

def energy_change(spins, Jx, Jy, i, j):
    """Calcula el cambio de energía si se flipa el spin (i, j)."""
    s_i = spins[i, j]
    ip, im = (i + 1) % L, (i - 1) % L
    jp, jm = (j + 1) % L, (j - 1) % L
    
    neighbors_sum = (
        Jx[i, j] * spins[i, jp] + Jx[i, jm] * spins[i, jm] +
        Jy[i, j] * spins[ip, j] + Jy[im, j] * spins[im, j]
    )
    return 2 * s_i * neighbors_sum

# ==========================================================
#              2. MODELO DE PILA DE ARENA (ASM)
# ==========================================================

def initialize_asm(L):
    """Inicializa la pila de arena 2D."""
    zc = 3 
    grains = np.random.randint(0, zc, size=(L, L))
    return grains, zc

def run_avalanche(grains, zc, start_i, start_j):
    """Ejecuta una avalancha y devuelve el patrón binario (True/False)."""
    grains[start_i, start_j] += 1
    topple_queue = [(start_i, start_j)]
    avalanche_pattern = np.zeros((L, L), dtype=bool)
    
    while topple_queue:
        i, j = topple_queue.pop(0)
        
        if grains[i, j] > zc:
            avalanche_pattern[i, j] = True
            grains[i, j] -= 4 
            
            # Distribuye a los 4 vecinos (PBC)
            for ni, nj in [((i + 1) % L, j), ((i - 1) % L, j), (i, (j + 1) % L), (i, (j - 1) % L)]:
                grains[ni, nj] += 1
                if grains[ni, nj] > zc and (ni, nj) not in topple_queue:
                    topple_queue.append((ni, nj))
                    
    S = np.sum(avalanche_pattern)
    return avalanche_pattern, S

# ==========================================================
#                 3. MÉTODOS DE OPTIMIZACIÓN
# ==========================================================

def apply_greedy_flip(spins, Jx, Jy, pattern):
    """Aplica el criterio voraz sobre un patrón de búsqueda (batch)."""
    flips_successful = False
    indices = np.argwhere(pattern)
    
    for i, j in indices:
        dE = energy_change(spins, Jx, Jy, i, j)
        if dE < 0:
            spins[i, j] *= -1 
            flips_successful = True
            
    return spins, flips_successful

def soc_search_step(spins, Jx, Jy, L, grains, zc):
    i_start, j_start = np.random.randint(0, L, size=2)
    pattern, S = run_avalanche(grains, zc, i_start, j_start)
    spins, success = apply_greedy_flip(spins, Jx, Jy, pattern)
    return spins, success, S

def random_local_search_step(spins, Jx, Jy, L):
    i, j = np.random.randint(0, L, size=2)
    dE = energy_change(spins, Jx, Jy, i, j)
    if dE < 0:
        spins[i, j] *= -1
        return spins, True
    return spins, False

def simulated_annealing_step(spins, Jx, Jy, L, T):
    i, j = np.random.randint(0, L, size=2)
    dE = energy_change(spins, Jx, Jy, i, j)
    
    if dE < 0 or np.random.rand() < np.exp(-dE / T):
        spins[i, j] *= -1
        return spins, True
    return spins, False

def random_dots_step(spins, Jx, Jy, L, S):
    """Aplica el criterio voraz sobre S puntos aleatorios."""
    if S == 0: return spins, False, 0
    pattern = np.zeros((L, L), dtype=bool)
    
    # Generar S índices aleatorios
    i_coords = np.random.randint(0, L, size=S)
    j_coords = np.random.randint(0, L, size=S)
    
    pattern[i_coords, j_coords] = True
    spins, success = apply_greedy_flip(spins, Jx, Jy, pattern)
    return spins, success, S


# ==========================================================
#                 4. FUNCIÓN DE EJECUCIÓN MAESTRA
# ==========================================================

def run_simulation(method_name, N_trials, max_steps, L, N_spins, S_bar):
    
    # Puntos de log en escala logarítmica
    log_steps = np.logspace(0, np.log10(MAX_FLIPS_PER_SPIN), num=20, endpoint=True, dtype=int) * N_SPINS
    log_steps = np.unique(log_steps[log_steps <= max_steps])
    flips_per_spin = log_steps / N_SPINS
    
    energy_history = np.zeros((N_trials, len(log_steps)))
    prob_success_history = np.zeros((N_trials, len(log_steps)))
    
    # Schedule de enfriamiento lineal (simple)
    T_start = 1.0 
    T_end = 0.001 
    cooling_schedule = np.linspace(T_start, T_end, max_steps)

    print(f"\n[{method_name}] Ejecutando {N_trials} ensayos. S_bar: {S_bar}")
    
    for trial in tqdm(range(N_trials), desc=f"Método {method_name}"):
        spins, Jx, Jy = initialize_ising_model(L)
        grains, zc = initialize_asm(L)
        
        log_index = 0
        total_successes = 0
        total_avalanches_size = []
        
        for step in range(1, max_steps + 1):
            
            success = False
            S_current = 1 

            if method_name == 'SOC Search':
                spins, success, S_current = soc_search_step(spins, Jx, Jy, L, grains, zc)
                total_avalanches_size.append(S_current)
            elif method_name == 'Simulated Annealing':
                T = cooling_schedule[step-1] 
                spins, success = simulated_annealing_step(spins, Jx, Jy, L, T)
            elif method_name == 'Random Local Search':
                spins, success = random_local_search_step(spins, Jx, Jy, L)
            elif method_name == 'Random Dots':
                spins, success, _ = random_dots_step(spins, Jx, Jy, L, S_bar)
            
            if success:
                total_successes += 1
                
            if log_index < len(log_steps) and step == log_steps[log_index]:
                current_energy = calculate_energy(spins, Jx, Jy)
                energy_per_spin = current_energy / N_spins
                prob_success = total_successes / step
                
                energy_history[trial, log_index] = energy_per_spin
                prob_success_history[trial, log_index] = prob_success
                
                log_index += 1

        if method_name == 'SOC Search' and total_avalanches_size:
            S_bar = int(np.mean(total_avalanches_size))
            
    # Resultados promediados
    mean_energy = np.mean(energy_history, axis=0)
    std_energy = np.std(energy_history, axis=0)
    mean_prob = np.mean(prob_success_history, axis=0)
    std_prob = np.std(prob_success_history, axis=0)
    
    return flips_per_spin, mean_energy, std_energy, mean_prob, std_prob, S_bar

# ==========================================================
#                   5. EJECUCIÓN Y VISUALIZACIÓN
# ==========================================================

methods = ['SOC Search', 'Simulated Annealing', 'Random Local Search', 'Random Dots']
results = {}
S_BAR_ISING = 20 

# Ejecución de todos los métodos (SOC primero para S_bar)
for method in methods:
    flips, mean_E, std_E, mean_P, std_P, S_BAR_ISING = run_simulation(method, N_TRIALS, MAX_STEPS, L, N_SPINS, S_BAR_ISING)
    results[method] = {'flips': flips, 'E_mean': mean_E, 'E_std': std_E, 'P_mean': mean_P, 'P_std': std_P}
    if method == 'SOC Search':
        print(f"[INFO] S_bar promedio calculado para Batch Methods: {S_BAR_ISING}")

# --- 6. Visualización (Gráfica de Energía) ---
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(results[method]['flips'], results[method]['E_mean'], label=method)
    plt.fill_between(results[method]['flips'], 
                     results[method]['E_mean'] - results[method]['E_std'], 
                     results[method]['E_mean'] + results[method]['E_std'], alpha=0.1)

plt.xscale('log')
plt.xlabel('Flips por Spin (F/S)')
plt.ylabel('Energía por Spin ($\\langle E/N \\rangle$)')
plt.title(f'Ising Spin Glass: Convergencia de Energía ({L}x{L})')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('Ising_Energy_Convergence.png')
plt.show()

# --- 7. Trazado de Probabilidad de Éxito ---
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(results[method]['flips'], results[method]['P_mean'], label=method)
    plt.fill_between(results[method]['flips'], 
                     results[method]['P_mean'] - results[method]['P_std'], 
                     results[method]['P_mean'] + results[method]['P_std'], alpha=0.1)

plt.xscale('log')
plt.xlabel('Flips por Spin (F/S)')
plt.ylabel('Probabilidad de Flip Exitoso ($\\langle P_{succ} \\rangle$)')
plt.title(f'Ising Spin Glass: Probabilidad de Éxito ({L}x{L})')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('Ising_Success_Probability.png')
plt.show()

print("\nResultados generados y guardados como PNGs.")