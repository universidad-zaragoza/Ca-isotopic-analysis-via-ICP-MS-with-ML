# -*- coding: utf-8 -*-
"""
#Created on Dec 11 12:25:41 2025

#@author: Javier Resano
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, d2_absolute_error_score
from error import error
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
import os
import joblib
from datetime import datetime


def save_ensemble(ensemble, directorio=None):
    """
    Saves an ensemble of sklearn models into a directory.

    Parameters:
    ensemble (list or dict): collection of models (e.g., [mlp1, mlp2, ...] or {"a": mlp1}).
    directorio (str, optional): path of the directory where they will be saved.
                                Defaults to './ensemble_YYYYMMDD_HHMMSS'
    Returns:
    str: path of the directory where the ensemble was saved.
    """

    if directorio is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directorio = f"ensemble_{timestamp}"

    os.makedirs(directorio, exist_ok=True)

    if isinstance(ensemble, dict):
        for nombre, modelo in ensemble.items():
            ruta = os.path.join(directorio, f"{nombre}.pkl")
            joblib.dump(modelo, ruta)
    elif isinstance(ensemble, list):
        for i, modelo in enumerate(ensemble):
            ruta = os.path.join(directorio, f"modelo_{i}.pkl")
            joblib.dump(modelo, ruta)
    else:
        raise TypeError("Ensemble must be a list or dictionary")

    return directorio


def ensemble_n_2_outputs(n, X_train, Y_train, X_test, Y_test, verbose=False, hidden_layer_sizes=(16, 16, 4), max_error=0.02, tag="MLP"):

    mlp = MLPRegressor(hidden_layer_sizes,
                       early_stopping=True,
                       max_iter=10000,
                       shuffle=True,
                       solver='adam',
                       verbose=verbose,
                       alpha=10e-3,
                       learning_rate_init=0.005)
    models = list()
    i = 0
    train_shape = [n, Y_train.shape[0], Y_train.shape[1]]
    test_shape = [n, Y_test.shape[0], Y_test.shape[1]]
    pred_train_vector = np.zeros(train_shape)
    pred_test_vector = np.zeros(test_shape)
    max_tries = 100
    Best_error = 1
    if verbose == True:
        print("Ensemble's size:")
        print(n)
    if n > 0:
        num_tries = 0
        while i < n:
            mlp.fit(X_train, Y_train)
            pred_train = mlp.predict(X_train)
            error_train = error(Y_train, pred_train, 'mse')
            if (error_train < Best_error):
                Best_error = error_train
                Best_mlp = mlp
                Best_pred_train = pred_train
                Best_pred_test = mlp.predict(X_test)
            num_tries = num_tries+1
            if (error_train < max_error) or (num_tries == max_tries):
                print("Iteración: ", i)
                pred_train_vector[i, :, :] = pred_train
                pred_test_vector[i, :, :] = mlp.predict(X_test)
                if (num_tries == max_tries):
                    print("No solution found with this max_error: ", max_error)
                    models.append(Best_mlp)
                    pred_train_vector[i, :, :] = Best_pred_train
                    pred_test_vector[i, :, :] = Best_pred_test
                else:
                    models.append(mlp)
                    pred_train_vector[i, :, :] = pred_train
                    pred_test_vector[i, :, :] = mlp.predict(X_test)
                if verbose == True:
                    print("MLP Train error: ", round(error_train, 3))
                i = i+1
                num_tries = 0
                Best_error = 1
    save_ensemble(models)
    return pred_train_vector, pred_test_vector, models


def positive_values(array):
    pos_array = [0 if x < 0 else x for x in array]
    return pos_array


def mlp_predictions_plot(mlp, X):
    Y = mlp.predict(X)
    # Representación
    plt.plot(X, Y, marker='o')  # Usa 'o' para mostrar los puntos
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gráfico de puntos conectados')
    plt.grid(True)
    plt.show()
    return X, Y


def compare_points(A: np.ndarray, B: np.ndarray):
    """
    Receives two numpy matrices of shape (n, d) representing sets of points.
    Plots points from A and B with different colors/symbols, connects each pair
    of points with a line, and returns a vector with the mean squared distance
    between each pair.

    Parameters:
    -----------
    A, B : np.ndarray
        Point matrices, both of shape (n, d).

    Returns:
    --------
    mse_vector : np.ndarray
        Vector of size n with the mean squared error per pair of points.
    """

    if A.shape != B.shape:
        raise ValueError("Las dos matrices deben tener la misma forma.")

    n, d = A.shape

    # Calcular distancias cuadráticas medias (MSE)
    mse_vector = np.mean((A - B)**2, axis=1)
    MAE_vector = np.mean(np.abs(A - B), axis=1)
    # Graficar
    plt.figure(figsize=(12, 12))

    if d == 2:
        # Puntos en 2D
        plt.scatter(A[:, 0], A[:, 1], c='blue', marker='o', label='Set A')
        plt.scatter(B[:, 0], B[:, 1], c='red', marker='s', label='Set B')

        # Conectar cada par con una línea
        for i in range(n):
            plt.plot([A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]],
                     'k--', linewidth=0.7)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title("Comparison between sets A and B")
        plt.axis("equal")
        plt.grid(True, linestyle='--', alpha=0.5)

    elif d == 3:
        # Puntos en 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(A[:, 0], A[:, 1], A[:, 2],
                   c='blue', marker='o', label='set A')
        ax.scatter(B[:, 0], B[:, 1], B[:, 2],
                   c='red', marker='^', label='set B')

        for i in range(n):
            ax.plot([A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]], [
                    A[i, 2], B[i, 2]], 'k--', linewidth=0.7)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title("Comparison between sets A and B")

    else:
        print("The function can only plot points in 2D or 3D.")

    plt.show()

    return mse_vector, MAE_vector


def plot_bars(x: np.ndarray, y: np.ndarray, n: int):
    """
    Divide el vector x en n intervalos entre su valor mínimo y máximo.
    Para cada intervalo, calcula la media de los valores de y en las
    posiciones correspondientes y genera una gráfica de barras.
    Además, muestra la cantidad de elementos en cada barra.

    Parámetros
    ----------
    x : np.ndarray
        Vector usado para definir los intervalos.
    y : np.ndarray
        Vector cuyos valores se promedian en cada intervalo de x.
    n : int
        Número de intervalos.

    Retorna
    -------
    medias : np.ndarray
        Vector con la media de y en cada intervalo.
    counts : np.ndarray
        Cantidad de elementos en cada intervalo.
    limites : np.ndarray
        Los límites de los intervalos generados.
    """

    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud.")

    # Definir límites de los intervalos
    limites = np.linspace(np.min(x), np.max(x), n+1)

    medias = []
    counts = []
    etiquetas = []

    for i in range(n):
        if i < n-1:
            mask = (x >= limites[i]) & (x < limites[i+1])
        else:
            mask = (x >= limites[i]) & (x <= limites[i+1])  # incluir el último

        valores_y = y[mask]
        count = valores_y.size
        media = np.mean(valores_y) if count > 0 else 0.0  # poner 0 si vacío

        medias.append(media)
        counts.append(count)

        # Etiqueta del intervalo con 5 decimales
        etiquetas.append(f"[{limites[i]:.5f}, {limites[i+1]:.5f}]")

    medias = np.array(medias)
    counts = np.array(counts)

    # Gráfica de barras
    plt.figure(figsize=(9, 5))
    bars = plt.bar(etiquetas, medias, color="skyblue", edgecolor="k")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Media de y")
    plt.xlabel("Intervalos de x")
    plt.title("Media de y por intervalos de x")

    # Agregar los conteos sobre cada barra
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, str(count),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return medias, counts, limites


def sort_and_plot(x: np.ndarray, y: np.ndarray):
    """
    Sorts vectors according to x values (ascending).
    Generates a plot where:
    - X Axis: sorted x values
    - Y Axis: cumulative mean and cumulative max of the first n values of y.

    Parameters
    ----------
    x : np.ndarray
        Vector used for sorting (X coordinates).
    y : np.ndarray
        Vector whose values are analyzed (Y coordinates).

    Returns
    -------
    x_ord : np.ndarray
        Sorted x vector.
    y_ord : np.ndarray
        Sorted y vector according to x.
    medias : np.ndarray
        Cumulative means of y_ord.
    maximos : np.ndarray
        Cumulative maximums of y_ord.
    """

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Ordenar según x
    orden = np.argsort(x)
    x_ord = x[orden]
    y_ord = y[orden]

    # Calcular medias y máximos acumulados
    n = len(y_ord)
    medias = np.array([np.mean(y_ord[:i+1]) for i in range(n)])
    maximos = np.array([np.max(y_ord[:i+1]) for i in range(n)])

    # Gráfica
    plt.figure(figsize=(8, 5))
    plt.plot(x_ord, medias, label="Cumulative Mean", color="blue", marker="o")
    plt.plot(x_ord, maximos, label="Cumulative Max", color="red", marker="s")
    plt.xlabel("X values (sorted)")
    plt.ylabel("Cumulative Y values")
    plt.title("Cumulative mean and max of y sorted by x")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return x_ord, y_ord, medias, maximos


def read_csv(name="5 mas altos.csv"):
    df_input = pd.read_csv(name, delimiter=';', dtype=np.float64)
    df_input = df_input.dropna(axis=1, how="all")
    input_data = df_input.to_numpy()
    input_data_without_bco = np.zeros((200, 605))
    input_data_without_bco_transposed = np.zeros((605, 200))
    input_data_without_noise = np.zeros((200, 605))
    for i in range(11):
        w_start = 1+i*60
        w_end = w_start + 5
        bco = input_data[:, w_start:w_end]
        bco_mean = np.average(bco, axis=1)
        for j in range(55):
            input_data_without_bco[:, i*55 + j] = input_data[:, i*60 + 6 + j]
            input_data_without_noise[:, i*55 +
                                     j] = input_data[:, i*60 + 6 + j] - bco_mean
        input_data_without_noise_transposed = input_data_without_noise.transpose()
        input_data_without_bco_transposed = input_data_without_bco.transpose()
    return input_data_without_bco_transposed, input_data_without_noise_transposed

# input_data_without_bco is the original input withoput the noise meassurments
# we extract the "white" meassurements from columns 1 to 6 and compute the average value
# Then, we substract this value to the next 55 columns (experiments from 1 to 11)


input_data_without_bco_transposed_5, input_data_without_noise_transposed_5 = read_csv(
    "5 mas altos.csv")

# Reading outputs
df_output = pd.read_csv("Solutions.csv", delimiter=';', dtype=np.float64)
output = df_output.to_numpy()
output_data = output[:, 1:3]
output_data_extended = np.zeros((605, 2))
for i in range(121):
    for j in range(5):
        output_data_extended[i*5+j, :] = output_data[i, 0:2]


# =============================================================================
# Experiment parameters
# =============================================================================

Predictions_MLP_vector = np.zeros((10, 605, 2))
Error_vector_MLP = np.zeros((605, 2))
ensemble_size = 50
verbose = False
hidden_layer_sizes = (64, 32, 32, 16)
max_error = 0.0003
number_inputs = 605
number_outputs = 2
Predictions_MLP_vector = np.zeros(
    (ensemble_size, number_inputs, number_outputs))
Error_vector_MLP = np.zeros((number_inputs, number_outputs))
Data_set_input = input_data_without_bco_transposed_5
Data_set_output = output_data_extended

# =============================================================================
# Experiment 1: Removing one of the inputs iteratively, and training an esemble of 50 MLPs with the remaining ones.
# We test each of the 121 model with the input removed
# Next training loop takes time. Store the results to prevent the need to repeat Training
# =============================================================================
# 121x50 iterations

for i in range(121):
    start = i * 5
    end = start + 5
    print(i)
    # Vector con los 5 consecutivos
    X_test = Data_set_input[start:end, :]
    Y_test = Data_set_output[start:end, :]
    # Vector con los 600 restantes
    X_train = np.delete(Data_set_input, np.s_[start:end], axis=0)
    Y_train = np.delete(Data_set_output, np.s_[start:end], axis=0)
    pred_train_full, Predictions_MLP_vector[:, start:end, :], mlp_full = ensemble_n_2_outputs(
        ensemble_size, X_train, Y_train, X_test, Y_test, verbose, hidden_layer_sizes, max_error)

np.save("Predictions_MLP_vector.npy",
        Predictions_MLP_vector)
# If previouslly computed uncoment next line to load the data
#Predictions_MLP_vector_loaded =np.load("Predictions_MLP_vector.npy")
Predictions_MLP = sum(Predictions_MLP_vector)/len(Predictions_MLP_vector)
Desviation_MLP = np.std(Predictions_MLP_vector, axis=0)
Error_vector = output_data_extended - Predictions_MLP
MSE_error = error(output_data_extended, Predictions_MLP, 'mse')
print("MSE_error: ", round(MSE_error, 6))
MAE_error = error(output_data_extended, Predictions_MLP, 'mae')
print("MAE_error: ", round(MAE_error, 6))
MSE_error = mean_squared_error(output_data_extended, Predictions_MLP)
print("MSE: ", MSE_error)
MAE_error = mean_absolute_error(output_data_extended, Predictions_MLP)
print("MAE: ", MAE_error)
mape_error = mean_absolute_percentage_error(
    output_data_extended, Predictions_MLP)
print("MAPE: ", mape_error)
d2 = d2_absolute_error_score(output_data_extended, Predictions_MLP)
print("D2: ", d2)
r2 = r2_score(output_data_extended, Predictions_MLP)
print("R2: ", r2)
# Results Figure
plt.figure(figsize=(12, 12))

# Set 1: red circles
plt.scatter(output_data_extended[:, 0], output_data_extended[:, 1],
            color='red', marker='o', label='Output_data')

# Set 2: tblue triangles
plt.scatter(Predictions_MLP[:, 0], Predictions_MLP[:, 1],
            color='blue', marker='^', label='Predictions')

plt.xlabel("cCa44/(cCa44+cCa40) ")
plt.ylabel("cCl37/(cCl37+cCl35)")
plt.legend()
plt.title("Predictions vs expected output MLP esemble of 4 trained with all the solutions but the one tested and the 200 inputs")
plt.grid(True)
plt.show()


MSE_vector, MAE_vector = compare_points(output_data_extended, Predictions_MLP)
# Mean Absolute Percentage Error (MAPE)
eps = 1e-8  # eps avoids division by zero
MAPE_vector = (np.abs(output_data_extended - Predictions_MLP) /
               (np.abs(output_data_extended) + eps))
matriz_correlacion = np.corrcoef(
    MAPE_vector[:, 0], Desviation_MLP[:, 0] / (np.abs(output_data_extended[:, 0]) + eps))
print(matriz_correlacion)
matriz_correlacion = np.corrcoef(
    MAPE_vector[:, 1], Desviation_MLP[:, 1] / (np.abs(output_data_extended[:, 1]) + eps))
print(matriz_correlacion)

d2 = d2_absolute_error_score(
    output_data_extended, Predictions_MLP, multioutput='uniform_average')
print(d2)


coeff_vartiation = Desviation_MLP / (np.abs(output_data_extended) + eps)
coeff_vartiation_merged = coeff_vartiation.reshape(-1)
# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(coeff_vartiation, MAPE_vector,
            color='red', marker='o', label='Relative deviation vs MAPE')

plt.xlabel("Coefficient of variation")
plt.ylabel("MAPE")
plt.legend()
plt.title("MAPE vs coefficient of variation")
plt.grid(True)
plt.show()

# Convert counts to percentages using weights
MAPE_vector_merged = MAPE_vector.reshape(-1)
# each value contributes to percentage
weights = np.ones_like(MAPE_vector_merged) / len(MAPE_vector_merged) * 100
bins = 20
x_range = (0, 1)
# Plot histogram
counts, bin_edges, patches = plt.hist(
    MAPE_vector_merged, bins=bins, range=x_range, weights=weights, edgecolor='black')

# Labels and title
plt.xlabel("MAPE")
plt.ylabel("Percentage (%)")
plt.title("Histogram for MAPE with percentages on y-axis and bar labels")
plt.ylim(0, counts.max()*1.1)  # add some space for labels
plt.show()

# Convert counts to percentages using weights

# each value contributes to percentage
weights = np.ones_like(coeff_vartiation_merged) / \
    len(coeff_vartiation_merged) * 100
bins = 20
x_range = (0, 1)
# Plot histogram
counts, bin_edges, patches = plt.hist(
    coeff_vartiation_merged, bins=bins, range=x_range, weights=weights, edgecolor='black')

# Labels and title
plt.xlabel("Coefficient of variation")
plt.ylabel("Percentage (%)")
plt.title(
    "Histogram for Coefficient of variation with percentages on y-axis and bar labels")
plt.ylim(0, counts.max()*1.1)  # add some space for labels
plt.show()


# Step 1: compute weights (inverse variance)
weights = 1 / Desviation_MLP**2

# Step 2: weighted mean (optional if you want the mean)
weighted_mean = np.sum(Predictions_MLP * weights) / np.sum(weights)

# Step 3: combined standard deviation
combined_std = 1 / np.sqrt(np.sum(weights))

print("Weighted mean:", weighted_mean)
print("Combined std:", combined_std)

weighted_mean_vector = np.ones([121, 2])
combined_std_vector = np.ones([121, 2])
for i in range(121):
    Predictions_MLP_5 = Predictions_MLP[i*5:i*5+5, :]
    Desviacion_MLP_5 = Desviation_MLP[i*5:i*5+5, :]
    weights = 1 / Desviacion_MLP_5**2
    weighted_mean_vector[i, 0] = np.sum(
        Predictions_MLP_5[:, 0] * weights[:, 0]) / np.sum(weights[:, 0])
    weighted_mean_vector[i, 1] = np.sum(
        Predictions_MLP_5[:, 1] * weights[:, 1]) / np.sum(weights[:, 1])
    combined_std_vector[i] = 1 / np.sqrt(np.sum(weights))


# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(output_data_extended[:, 0], output_data_extended[:, 1],
            color='red', marker='o', label='Output_data')

# Conjunto 2: triángulos azules
plt.scatter(weighted_mean_vector[:, 0], weighted_mean_vector[:, 1],
            color='blue', marker='^', label='Predictions')

plt.xlabel("cCa44/(cCa44+cCa40) ")
plt.ylabel("cCl37/(cCl37+cCl35)")
plt.legend()
plt.title("Predictions vs expected output MLP esemble of 50 trained with all the solutions but the one tested and the 200 inputs")
plt.grid(True)
plt.show()

mse_combined = mean_squared_error(output_data, weighted_mean_vector)
print("MSE: ", mse_combined)
mae_combined = mean_absolute_error(output_data, weighted_mean_vector)
print("MAE: ", mae_combined)
mape_combined = mean_absolute_percentage_error(
    output_data, weighted_mean_vector)
print("MAPE: ", mape_combined)
d2_combined = d2_absolute_error_score(output_data, weighted_mean_vector)
print("D2: ", d2_combined)

compare_points(output_data, weighted_mean_vector)
###########################################################################

weights = np.ones_like(MAPE_vector) / len(MAPE_vector) * \
    100  # convert counts to %

plt.hist(MAPE_vector, bins=20, range=(0, 1), weights=weights)
plt.xlabel("MAPE")
plt.ylabel("Frequency(%)")
plt.title("Histogram for MAPE")
plt.show()


matriz_correlacion = np.corrcoef(MAPE_vector[:, 0], Desviation_MLP[:, 0])
print(matriz_correlacion)
matriz_correlacion = np.corrcoef(MAPE_vector[:, 1], Desviation_MLP[:, 1])
print(matriz_correlacion)


matriz_correlacion = np.corrcoef(MAE_vector, Desviation_MLP[:, 0])
print(matriz_correlacion)
matriz_correlacion = np.corrcoef(MAE_vector, Desviation_MLP[:, 1])
print(matriz_correlacion)
Suma_Desviation = Desviation_MLP[:, 0] + Desviation_MLP[:, 1]
matriz_correlacion = np.corrcoef(MAE_vector, Suma_Desviation)
print(matriz_correlacion)
medias, counts, limites = plot_bars(Suma_Desviation, MAE_vector, 50)
Suma_Desviations_ord, MAE_ord, medias, maximos = sort_and_plot(
    Suma_Desviation, MAE_vector)
Desviations_0_ord, MAE_0_ord, medias, maximos = sort_and_plot(
    Desviation_MLP[:, 0], MAE_vector)
# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(Suma_Desviation[0:605], MAE_vector[0:605],
            color='red', marker='o', label='Output_data')

plt.xlabel("Std Deviation")
plt.ylabel("MAE")
plt.legend()
plt.title("MAE vs Std deviation")
plt.grid(True)
plt.show()
###############################################################################
# Combine predictions form the five experiments into 1 prediction
###############################################################################
Predictions_MLP_avg = np.zeros((int(number_inputs/5), 2))
Desviacion_MLP_5 = np.zeros((int(number_inputs/5), 2))

for i in range(121):
    Predictions_MLP_avg[i, :] = np.mean(Predictions_MLP[i*5:i*5+5, :], axis=0)
    Desviacion_MLP_5[i, 0] = np.std(Predictions_MLP_vector[:, i*5:i*5+5, 0])
    Desviacion_MLP_5[i, 1] = np.std(Predictions_MLP_vector[:, i*5:i*5+5, 1])
    
MSE_vector_avg, MAE_vector_avg = compare_points(
    output_data, Predictions_MLP_avg)
Error_vector_avg = output_data - Predictions_MLP_avg
mse_combined = mean_squared_error(output_data, Predictions_MLP_avg)
print("MSE: ", mse_combined)
mae_combined = mean_absolute_error(output_data, Predictions_MLP_avg)
print("MAE: ", mae_combined)
mape_combined = mean_absolute_percentage_error(
    output_data, Predictions_MLP_avg)
print("MAPE: ", mape_combined)
d2_combined = d2_absolute_error_score(output_data, Predictions_MLP_avg)
print("D2: ", d2_combined)
r2_combined = r2_score(output_data, Predictions_MLP_avg)
print("R2: ", r2_combined)

matriz_correlacion = np.corrcoef(MAE_vector_avg, Desviacion_MLP_5[:, 0])
print(matriz_correlacion)
matriz_correlacion = np.corrcoef(MAE_vector_avg, Desviacion_MLP_5[:, 1])
print(matriz_correlacion)
Suma_Desviacion_MLP_5 = Desviacion_MLP_5[:, 0] + Desviacion_MLP_5[:, 1]
matriz_correlacion = np.corrcoef(MAE_vector_avg, Suma_Desviacion_MLP_5)
print(matriz_correlacion)

medias, counts, limites = plot_bars(MAE_vector_avg, Desviacion_MLP_5[:, 0], 50)
medias, counts, limites = plot_bars(MAE_vector_avg, Desviacion_MLP_5[:, 1], 50)

medias, counts, limites = plot_bars(Desviacion_MLP_5[:, 0], MAE_vector_avg, 50)
medias, counts, limites = plot_bars(Desviacion_MLP_5[:, 1], MAE_vector_avg, 50)

medias, counts, limites = plot_bars(Suma_Desviacion_MLP_5, MAE_vector_avg, 50)


Suma_Desviaciones_ord_avg, MAE_ord_avg, medias_avg, maximos_avg = sort_and_plot(
    Suma_Desviacion_MLP_5, MAE_vector_avg)

# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(Suma_Desviacion_MLP_5[0:120], MAE_vector_avg[0:120],
            color='red', marker='o', label='Output_data')

plt.xlabel("Std Deviation")
plt.ylabel("MAE")
plt.legend()
plt.title("MAE vs Std deviation")
plt.grid(True)
plt.show()


# =============================================================================
# Experiment 2: normalizing each row from 0 to 1
# =============================================================================

def normalize_rows_0_1(arr):
    """
    Normalize each row of a 2D NumPy array to [0, 1].

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (605, 200) (or any 2D shape).

    Returns
    -------
    np.ndarray
        Row-wise normalized array with the same shape.
    """
    row_min = arr.min(axis=1, keepdims=True)
    row_max = arr.max(axis=1, keepdims=True)

    # Avoid division by zero for constant rows
    denom = row_max - row_min
    denom[denom == 0] = 1

    return (arr - row_min) / denom


Predictions_MLP_vector = np.zeros((10, 605, 2))
Error_vector_MLP = np.zeros((605, 2))
ensemble_size = 50
verbose = False
hidden_layer_sizes = (64, 32, 32, 16)
max_error = 0.0003
number_inputs = 605
number_outputs = 2
Predictions_MLP_vector = np.zeros(
    (ensemble_size, number_inputs, number_outputs))
Error_vector_MLP = np.zeros((number_inputs, number_outputs))
Data_set_input = normalize_rows_0_1(input_data_without_bco_transposed_5)
Data_set_output = output_data_extended

#################################################### 
# Experiment 2: Removing one of the inputs iteratively, and training an esemble of 50 MLPs with the remaining ones.
# We test each of the 121 model with the input removed
# Next training loop takes time. Store the results to prevent the need to repeat training
# 121x50 iterations

for i in range(121):
    start = i * 5
    end = start + 5
    print(i)
    # Vector con los 5 consecutivos
    X_test = Data_set_input[start:end, :]
    Y_test = Data_set_output[start:end, :]
    # Vector con los 600 restantes
    X_train = np.delete(Data_set_input, np.s_[start:end], axis=0)
    Y_train = np.delete(Data_set_output, np.s_[start:end], axis=0)
    pred_train_full, Predictions_MLP_vector[:, start:end, :], mlp_full = ensemble_n_2_outputs(
        ensemble_size, X_train, Y_train, X_test, Y_test, verbose, hidden_layer_sizes, max_error)

np.save("Predictions_MLP_vector_5 espectros_sin_filtrar_anomalos_norm_26_12.npy",
        Predictions_MLP_vector)
#Predictions_MLP_vector_loaded =np.load("Predictions_MLP_vector.npy")
Predictions_MLP = sum(Predictions_MLP_vector)/len(Predictions_MLP_vector)
Desviation_MLP = np.std(Predictions_MLP_vector, axis=0)
# idea: calcular la varianza de las 5 predicciones
Error_vector = output_data_extended - Predictions_MLP
MSE_error = error(output_data_extended, Predictions_MLP, 'mse')
print("MSE_error: ", round(MSE_error, 6))
MAE_error = error(output_data_extended, Predictions_MLP, 'mae')
print("MAE_error: ", round(MAE_error, 6))

MSE_error = mean_squared_error(output_data_extended, Predictions_MLP)
print("MSE: ", MSE_error)
MAE_error = mean_absolute_error(output_data_extended, Predictions_MLP)
print("MAE: ", MAE_error)
mape_error = mean_absolute_percentage_error(
    output_data_extended, Predictions_MLP)
print("MAPE: ", mape_error)
d2 = d2_absolute_error_score(output_data_extended, Predictions_MLP)
print("D2: ", d2)
r2 = r2_score(output_data_extended, Predictions_MLP)
print("R2: ", r2)
r2_0 = r2_score(output_data_extended[:, 0], Predictions_MLP[:, 0])
print("R2_Ca: ", r2_0)
r2_1 = r2_score(output_data_extended[:, 1], Predictions_MLP[:, 1])
print("R2_Cl: ", r2_1)

# Step 1: compute weights (inverse variance)
weights = 1 / Desviation_MLP**2

# Step 2: weighted mean (optional if you want the mean)
weighted_mean = np.sum(Predictions_MLP * weights) / np.sum(weights)

# Step 3: combined standard deviation
combined_std = 1 / np.sqrt(np.sum(weights))

print("Weighted mean:", weighted_mean)
print("Combined std:", combined_std)

weighted_mean_vector = np.ones([121, 2])
combined_std_vector = np.ones([121, 2])
for i in range(121):
    Predictions_MLP_5 = Predictions_MLP[i*5:i*5+5, :]
    Desviacion_MLP_5 = Desviation_MLP[i*5:i*5+5, :]
    weights = 1 / Desviacion_MLP_5**2
    weighted_mean_vector[i, 0] = np.sum(
        Predictions_MLP_5[:, 0] * weights[:, 0]) / np.sum(weights[:, 0])
    weighted_mean_vector[i, 1] = np.sum(
        Predictions_MLP_5[:, 1] * weights[:, 1]) / np.sum(weights[:, 1])
    combined_std_vector[i] = 1 / np.sqrt(np.sum(weights))


# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(output_data_extended[:, 0], output_data_extended[:, 1],
            color='red', marker='o', label='Output_data')

# Conjunto 2: triángulos azules
plt.scatter(weighted_mean_vector[:, 0], weighted_mean_vector[:, 1],
            color='blue', marker='^', label='Predictions')

np.savetxt(
    "weighted_mean_predictions.csv",
    weighted_mean_vector,
    header="44Ca/40Ca, 37Cl/35Cl",
    delimiter=",",
    fmt="%.6f"   # number of decimal places
)
np.savetxt(
    "combined_std_vector.csv",
    combined_std_vector,
    header="44Ca/40Ca, 37Cl/35Cl",
    delimiter=",",
    fmt="%.6f"   # number of decimal places
)


plt.xlabel("cCa44/(cCa44+cCa40) ")
plt.ylabel("cCl37/(cCl37+cCl35)")
plt.legend()
plt.title("Predictions vs expected output MLP esemble of 50 trained with all the solutions but the one tested and the 200 inputs")
plt.grid(True)
plt.show()

mse_combined = mean_squared_error(output_data, weighted_mean_vector)
print("MSE: ", mse_combined)
mae_combined = mean_absolute_error(output_data, weighted_mean_vector)
print("MAE: ", mae_combined)
mape_combined = mean_absolute_percentage_error(
    output_data, weighted_mean_vector)
print("MAPE: ", mape_combined)
d2_combined = d2_absolute_error_score(output_data, weighted_mean_vector)
print("D2: ", d2_combined)
r2_combined = r2_score(output_data, weighted_mean_vector)
print("R2: ", r2_combined)
compare_points(output_data, weighted_mean_vector)

MAPE_vector = (np.abs(output_data - weighted_mean_vector) /
               (np.abs(output_data) + eps))
coeff_vartiation = combined_std_vector / (np.abs(output_data) + eps)
matriz_correlacion = np.corrcoef(MAPE_vector[:, 0], coeff_vartiation[:, 0])
print(matriz_correlacion)
matriz_correlacion = np.corrcoef(MAPE_vector[:, 1], coeff_vartiation[:, 1])
print(matriz_correlacion)

# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(coeff_vartiation, MAPE_vector,
            color='red', marker='o', label='Predictions')

plt.xlabel("Coeff vartiation")
plt.ylabel("MAPE")
plt.legend()
plt.title("MAPE vs Coeff vartiation")
plt.grid(True)
plt.show()

# One-dimension MAPE

mape_overall = np.zeros([121])
for i in range(121):
    mape_overall[i] = np.mean(
        np.abs((output_data[i, :] - weighted_mean_vector[i, :]
                ) / (output_data[i, :] + eps))
    ) * 100
coeff_variation_avg = (coeff_vartiation[:, 0]+coeff_vartiation[:, 1])/2
matriz_correlacion = np.corrcoef(mape_overall, coeff_variation_avg)
print(matriz_correlacion)

# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(coeff_variation_avg, mape_overall,
            color='red', marker='o', label='Predictions')

plt.xlabel("Coeff vartiation")
plt.ylabel("MAPE")
plt.legend()
plt.title("MAPE vs Coeff vartiation")
plt.grid(True)
plt.show()


###############################################################################
# =============================================================================
# Experiment 3: Using MLP with all the initial inputs, and testing with the new inputs
# =============================================================================

# entradas nuevas
# Nota: para evitar errores hay que eliminar la segunda fila. Cambiar el formato de las longitudes a números (copiando el formato de otros datos) y cambiar las comas por puntos
df_input_5_nuevas_v2 = pd.read_csv(
    "5_nuevas_hp_v2.csv", delimiter=';', dtype=np.float64)
df_input_5_nuevas = pd.read_csv(
    "5_nuevas_v2.csv", delimiter=';', dtype=np.float64)
df_input_5_nuevas = df_input_5_nuevas.dropna(axis=1, how="all")
input_data_5_nuevas = df_input_5_nuevas.to_numpy()
df_input_5_nuevas_v2 = df_input_5_nuevas_v2.dropna(axis=1, how="all")
input_data_5_nuevas_v2 = df_input_5_nuevas_v2.to_numpy()
diff = df_input_5_nuevas_v2-df_input_5_nuevas
diff_abs = np.sum(diff)
input_data_5_nuevas_without_bco = np.zeros((200, 130))
input_data_5_nuevas_without_bco_transposed = np.zeros((130, 200))
input_data_5_nuevas_without_noise = np.zeros((200, 130))
input_data_5_nuevas_without_noise_transposed = np.zeros((130, 200))


for i in range(2):
    w_start = 1+i*60
    w_end = w_start + 5
    bco_5_nuevas = input_data_5_nuevas_v2[:, w_start:w_end]
    bco_mean_5 = np.average(bco_5_nuevas, axis=1)
    for j in range(65):
        input_data_5_nuevas_without_bco[:, i*65 +
                                        j] = input_data_5_nuevas_v2[:, i*70 + 6 + j]
        input_data_5_nuevas_without_noise[:, i*65 +
                                          j] = input_data_5_nuevas_v2[:, i*70 + 6 + j] - bco_mean_5
    input_data_5_nuevas_without_noise_transposed = input_data_5_nuevas_without_noise.transpose()
    input_data_5_nuevas_without_bco_transposed = input_data_5_nuevas_without_bco.transpose()


Predictions_MLP_vector = np.zeros((10, 605, 2))
Error_vector_MLP = np.zeros((605, 2))
ensemble_size = 50
verbose = False
hidden_layer_sizes = (64, 32, 32, 16)
max_error = 0.00015
number_inputs = 605
number_outputs = 2
Predictions_MLP_vector = np.zeros(
    (ensemble_size, number_inputs, number_outputs))
Error_vector_MLP = np.zeros((number_inputs, number_outputs))
Data_set_input = normalize_rows_0_1(input_data_without_bco_transposed_5)
Data_set_output = output_data_extended

# New data used for test
X_test = normalize_rows_0_1(input_data_5_nuevas_without_noise_transposed)
Y_test = np.zeros([130, 2])
# Reading outputs
df_output = pd.read_csv("Solutions_nuevas.csv",
                        delimiter=';', dtype=np.float64)
output_nuevas = df_output.to_numpy()
output_data_nuevas = output_nuevas[:, 1:3]
output_data_nuevas_extended = np.zeros((26*5, 2))
for i in range(26):
    for j in range(5):
        output_data_nuevas_extended[i*5+j, :] = output_data_nuevas[i, 0:2]
Y_test = output_data_nuevas_extended
# Original data used for train
X_train = Data_set_input
Y_train = Data_set_output
pred_train_full, Predictions_MLP_5_nuevas_vector, mlp_full = ensemble_n_2_outputs(
    ensemble_size, X_train, Y_train, X_test, Y_test, verbose, hidden_layer_sizes, max_error)


# Reshape: we have 5x50 predictions for each value
Predictions_MLP_5_nuevas_vector_reshaped_v1 = Predictions_MLP_5_nuevas_vector.reshape(
    50, 26, 5, 2)  # → shape (50, 5, 26, 2)
# Step 2: Move the repetition axis (5) into the model axis (50)
Predictions_MLP_5_nuevas_vector_reshaped_v1_mean = Predictions_MLP_5_nuevas_vector_reshaped_v1.mean(
    axis=0)
Predictions_MLP_5_nuevas_vector_reshaped_v1_std = Predictions_MLP_5_nuevas_vector_reshaped_v1.std(
    axis=0)

weighted_mean_vector_news = np.ones([26, 2])
combined_std_vector_news = np.ones([26, 2])
for i in range(26):
    Predictions_MLP_5 = Predictions_MLP_5_nuevas_vector_reshaped_v1_mean[i, :, :]
    Desviacion_MLP_5 = Predictions_MLP_5_nuevas_vector_reshaped_v1_std[i, :, :]
    weights = 1 / Desviacion_MLP_5**2
    weighted_mean_vector_news[i, 0] = np.sum(
        Predictions_MLP_5[:, 0] * weights[:, 0]) / np.sum(weights[:, 0])
    weighted_mean_vector_news[i, 1] = np.sum(
        Predictions_MLP_5[:, 1] * weights[:, 1]) / np.sum(weights[:, 1])
    combined_std_vector_news[i] = 1 / np.sqrt(np.sum(weights))

Predictions_MLP_5_nuevas_vector_reshaped_v1_variation_coefficient = Predictions_MLP_5_nuevas_vector_reshaped_v1_std/np.abs(Predictions_MLP_5_nuevas_vector_reshaped_v1_mean)


Predictions_MLP_5_nuevas = sum(Predictions_MLP_5_nuevas_vector)/len(Predictions_MLP_5_nuevas_vector)    
Desviation_MLP_5_nuevas = np.std(Predictions_MLP_5_nuevas_vector, axis=0)
# Vamos a generar una única predicción para las 5 muestras
number_inputs = 130
Predictions_MLP_avg_5_nuevas = np.zeros((int(number_inputs/5), 2))
Desviacion_MLP_5_5_nuevas = np.zeros((int(number_inputs/5), 2))
Predictions_MLP_avg_5_nuevas = Predictions_MLP_5_nuevas.reshape(26, 5, 2).mean(axis=1)
Desviacion_MLP_5_5_nuevas[i,0] = np.std(Predictions_MLP_5_nuevas.reshape(26, 5, 2)[:,i*5:i*5+5,0])
Desviacion_MLP_5_5_nuevas[i,1] = np.std(Predictions_MLP_5_nuevas_vector[:,i*5:i*5+5,1])

for i in range(26):
    Predictions_MLP_avg_5_nuevas[i,:]= np.mean(Predictions_MLP_5_nuevas[i*5:i*5+5,:], axis=0)
    Desviacion_MLP_5_5_nuevas[i,0] = np.std(Predictions_MLP_5_nuevas_vector[:,i*5:i*5+5,0])
    Desviacion_MLP_5_5_nuevas[i,1] = np.std(Predictions_MLP_5_nuevas_vector[:,i*5:i*5+5,1])
    

# Analizing results
SE_vector_avg_nuevas, MAE_vector_avg_nuevas = compare_points(
    output_data_nuevas_extended, Predictions_MLP_5_nuevas_vector_reshaped_v1_mean.reshape(26*5, 2))
Error_vector_avg_nuevas = output_data_nuevas_extended - \
    Predictions_MLP_5_nuevas_vector_reshaped_v1_mean.reshape(26*5, 2)
MSE_error_nuevas = error(
    output_data_nuevas, Predictions_MLP_avg_5_nuevas, 'mse')
print("MSE_error: ", round(MSE_error_nuevas, 6))
MAE_error_nuevas = error(
    output_data_nuevas, Predictions_MLP_avg_5_nuevas, 'mae')
print("MAE_error: ", round(MAE_error_nuevas, 6))
MAE_error_nuevas_007 = error(
    output_data_nuevas[0:13, :], Predictions_MLP_avg_5_nuevas[0:13, :], 'mae')
MAE_error_nuevas_242 = error(
    output_data_nuevas[13:26, :], Predictions_MLP_avg_5_nuevas[13:26, :], 'mae')
print("MAE_error 0.007: ", round(MAE_error_nuevas_007, 6))
print("MAE_error 0.242: ", round(MAE_error_nuevas_242, 6))
MAE_error_nuevas_Ca = error(
    output_data_nuevas[:, 0], Predictions_MLP_avg_5_nuevas[:, 0], 'mae')
MAE_error_nuevas_Cl = error(
    output_data_nuevas[:, 1], Predictions_MLP_avg_5_nuevas[:, 1], 'mae')
print("MAE_error Ca: ", round(MAE_error_nuevas_Ca, 6))
print("MAE_error Cl: ", round(MAE_error_nuevas_Cl, 6))


matriz_correlacion_nuevas = np.corrcoef(
    MAE_vector_avg_nuevas, Desviacion_MLP_5_5_nuevas[:, 0])
print(matriz_correlacion_nuevas)
matriz_correlacion_nuevas = np.corrcoef(
    MAE_vector_avg_nuevas, Desviacion_MLP_5_5_nuevas[:, 1])
print(matriz_correlacion_nuevas)
Suma_Desviacion_MLP_5_nuevas = Desviacion_MLP_5_5_nuevas[:,
                                                         0] + Desviacion_MLP_5_5_nuevas[:, 1]
matriz_correlacion_nuevas = np.corrcoef(
    MAE_vector_avg_nuevas, Suma_Desviacion_MLP_5_nuevas)
print(matriz_correlacion_nuevas)

medias, counts, limites = plot_bars(
    MAE_vector_avg_nuevas, Desviacion_MLP_5_5_nuevas[:, 0], 50)
medias, counts, limites = plot_bars(
    MAE_vector_avg_nuevas, Desviacion_MLP_5_5_nuevas[:, 1], 50)

medias, counts, limites = plot_bars(
    Desviacion_MLP_5_5_nuevas[:, 0], MAE_vector_avg_nuevas, 50)
medias, counts, limites = plot_bars(
    Desviacion_MLP_5_5_nuevas[:, 1], MAE_vector_avg_nuevas, 50)

medias, counts, limites = plot_bars(
    Suma_Desviacion_MLP_5_nuevas, MAE_vector_avg_nuevas, 50)


Suma_Desviaciones_ord_avg, MAE_ord_avg, medias_avg, maximos_avg = sort_and_plot(
    Suma_Desviacion_MLP_5_nuevas, MAE_vector_avg_nuevas)

# Results Figure
plt.figure(figsize=(12, 12))

# Conjunto 1: círculos rojos
plt.scatter(Suma_Desviacion_MLP_5[0:120], MAE_vector_avg[0:120],
            color='red', marker='o', label='Output_data')

plt.xlabel("Std Deviation")
plt.ylabel("MAE")
plt.legend()
plt.title("MAE vs Std deviation")
plt.grid(True)
plt.show()

