if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import polymer_mmc as polym
    
    # print of a typical move

    total_moves=15

    configurations=np.zeros((total_moves, polym.n_monomers, 2))
    configurations[0]=polym.monomers_initial_conf

    for i in range(1,total_moves):
        configurations[i]=polym.polymer_displacement(configurations[i-1])

    conf=configurations[total_moves-1]

    # Move plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='datalim')

    for i, point in enumerate(conf):
        circle = plt.Circle((point[0], point[1]), 0.5, color='dodgerblue', alpha=1,linewidth=1.5)
        ax.add_patch(circle)
        if i > 0:
            prev_point = conf[i - 1]
            ax.plot([prev_point[0], point[0]], [prev_point[1], point[1]], color='dodgerblue', linewidth=2)

    ax.set_xlim(min(p[0] for p in conf) - 1, max(p[0] for p in conf) + 1)
    ax.set_ylim(min(p[1] for p in conf) - 1, max(p[1] for p in conf) + 1)
    ax.axhline(y=-polym.monomer_radius, color='red', linestyle='-', linewidth=2, label="Wall")
    ax.axhline(y=polym.energy_threshold, color='purple', linestyle=':', linewidth=1.5, label="Energy Threshold")
    ax.set_xlabel("X-axis", fontsize=12, labelpad=10, color='darkblue')
    ax.set_ylabel("Y-axis", fontsize=12, labelpad=10, color='darkblue')
    ax.set_title(f"Move Configuration after {total_moves} Moves", fontsize=14, color='darkblue', pad=15)

    plt.show()

    # thermalization phase

    x=np.arange(0,polym.time_max,1)

    beta_list =np.linspace(5, 50, 1)

    ee2_results = np.zeros((len(beta_list), polym.time_max))
    end_height_results = np.zeros((len(beta_list), polym.time_max))
    energy_evolution_results = np.zeros((len(beta_list), polym.time_max))

    for k, beta in enumerate(beta_list):
        ee2_results[k], end_height_results[k], energy_evolution_results[k] = polym.thermalization(beta)

    for i, beta in enumerate(beta_list):
        plt.plot(x, ee2_results[i], label=f'beta={beta_list[i]}')
    plt.title(f"End to end evolution at different betas")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

    for i, beta in enumerate(beta_list):
        plt.plot(x, end_height_results[i], label=f'beta={beta_list[i]}')
    plt.title(f"End heigth at different betas")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

    for i, beta in enumerate(beta_list):
        plt.plot(x, energy_evolution_results[i], label=f'beta={beta_list[i]}')
    plt.title(f"Energy evolution at different betas")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

    for i, beta in enumerate(beta_list):
        plt.hist(energy_evolution_results[i], bins=30, density=True, alpha=0.4,label=f'beta={beta_list[i]}')
    plt.title(f"Energy distribution at different betas")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()            