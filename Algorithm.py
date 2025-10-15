import numpy as np
import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')


class HybridFireflyParticleSwarmOptimization:
    """Hybrid Firefly Particle Swarm Optimization Algorithm for Feature Selection"""

    def __init__(self, n_particles=20, max_iterations=100, cognitive_factor=1.4,
                 social_factor=1.4, w_initial=0.9, w_final=0.4, alpha=0.5,
                 beta0=0.2, gamma=0.1, random_state=42):
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.cognitive_factor = cognitive_factor
        self.social_factor = social_factor
        self.w_initial = w_initial
        self.w_final = w_final
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.random_state = random_state

        np.random.seed(random_state)

        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = None
        self.fitness_history = []

        self.X_train = None
        self.y_train = None
        self.n_features = None

    def _initialize_population(self):
        self.positions = np.random.uniform(0, 1, (self.n_particles, self.n_features))
        self.velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_features))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def _binarize_position(self, position):
        binary_position = (position > 0.5).astype(int)
        if np.sum(binary_position) == 0:
            binary_position[np.random.randint(0, len(binary_position))] = 1
        return binary_position

    def _fitness_function(self, binary_position, X_val, y_val):
        selected_features = np.where(binary_position == 1)[0]
        if len(selected_features) == 0:
            return 1.0
        X_train_selected = self.X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_selected, self.y_train)
        y_pred = knn.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        return 1 - accuracy

    def _calculate_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def _update_inertia_weight(self, iteration):
        return self.w_initial - ((self.w_initial - self.w_final) / self.max_iterations) * iteration

    def _firefly_update(self, i, iteration):
        distance = self._calculate_distance(self.positions[i], self.global_best_position)
        attractiveness = self.beta0 * np.exp(-self.gamma * distance ** 2)
        random_term = self.alpha * np.random.normal(0, 1, self.n_features)
        new_position = (self.positions[i] +
                        attractiveness * (self.global_best_position - self.positions[i]) +
                        random_term)
        return new_position

    def _pso_update(self, i, iteration):
        w = self._update_inertia_weight(iteration)
        r1, r2 = np.random.random(2)
        cognitive_component = (self.cognitive_factor * r1 *
                               (self.personal_best_positions[i] - self.positions[i]))
        social_component = (self.social_factor * r2 *
                            (self.global_best_position - self.positions[i]))
        self.velocities[i] = (w * self.velocities[i] +
                              cognitive_component + social_component)
        new_position = self.positions[i] + self.velocities[i]
        return new_position

    def _hybrid_update(self, i, iteration, X_val, y_val):
        current_binary = self._binarize_position(self.positions[i])
        current_fitness = self._fitness_function(current_binary, X_val, y_val)

        if current_fitness <= self.global_best_score:
            if self.global_best_position is not None:
                pos_temp = self.positions[i].copy()
                new_position = self._firefly_update(i, iteration)
                self.velocities[i] = new_position - pos_temp
            else:
                new_position = self.positions[i]
        else:
            new_position = self._pso_update(i, iteration)

        return np.clip(new_position, 0, 1)

    def fit(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.n_features = X_train.shape[1]
        self._initialize_population()

        for iteration in range(self.max_iterations):
            for i in range(self.n_particles):
                self.positions[i] = self._hybrid_update(i, iteration, X_val, y_val)
                binary_position = self._binarize_position(self.positions[i])
                fitness = self._fitness_function(binary_position, X_val, y_val)

                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i].copy()

            self.fitness_history.append(self.global_best_score)
            if iteration % 10 == 0:
                selected_features = np.sum(self._binarize_position(self.global_best_position))
                accuracy = 1 - self.global_best_score
                print(f"KNN: Iteration {iteration}: Best Accuracy = {accuracy:.4f}, "
                      f"Features Selected = {selected_features}")

    def get_selected_features(self):
        binary_position = self._binarize_position(self.global_best_position)
        return np.where(binary_position == 1)[0]

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.fitness_history)),
                 [1 - f for f in self.fitness_history])
        plt.xlabel('Iteration')
        plt.ylabel('Best Accuracy')
        plt.title('HFPSO Convergence Curve')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Load Arcene
    X_train = pd.read_csv("arcene_train.data", delim_whitespace=True, header=None).values
    y_train = pd.read_csv("arcene_train.labels", header=None).values.ravel()
    y_train = np.where(y_train == -1, 0, 1)

    X_valid = pd.read_csv("arcene_valid.data", delim_whitespace=True, header=None).values
    y_valid = pd.read_csv("arcene_valid.labels", header=None).values.ravel()
    y_valid = np.where(y_valid == -1, 0, 1)

    print("Starting HFPSO on Arcene Training Data...")
    hfpso = HybridFireflyParticleSwarmOptimization(n_particles=10, max_iterations=50)
    hfpso.fit(X_train, y_train, X_valid, y_valid)

    selected_features = hfpso.get_selected_features()
    print(f"\nSelected Features ({len(selected_features)}): {selected_features}")

    # Evaluate metrics on validation set
    X_valid_selected = X_valid[:, selected_features]
    X_train_selected = X_train[:, selected_features]

    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_valid_selected = scaler.transform(X_valid_selected)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_selected, y_train)
    y_valid_pred = knn.predict(X_valid_selected)

    print("\n--- Validation Set Performance ---")
    print("Accuracy:", accuracy_score(y_valid, y_valid_pred))
    print("Precision:", precision_score(y_valid, y_valid_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_valid, y_valid_pred, average='weighted', zero_division=0))
    print("F1-Score:", f1_score(y_valid, y_valid_pred, average='weighted', zero_division=0))

    hfpso.plot_convergence()
