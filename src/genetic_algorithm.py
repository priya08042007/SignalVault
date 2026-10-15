"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: genetic_algorithm.py
Description: Reproducible, lightweight Genetic Algorithm for feature selection
with binary chromosome encoding, parsimony penalty, and generational tracking.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from src.eda import SOC_COLORS, SOC_PLOT_LAYOUT


class GeneticFeatureSelector:
    """
    Genetic Algorithm optimizer that searches the combinatorial 2^D space
    of original input features to discover minimal, high-performing feature subsets.
    """

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        feature_penalty: float = 0.05,
        tournament_size: int = 3,
        random_state: int = 42
    ):
        self.pop_size = max(6, population_size)
        self.generations = max(2, generations)
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.penalty = feature_penalty
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.history: List[Dict[str, Any]] = []
        self.best_chromosome: Optional[np.ndarray] = None
        self.best_fitness: float = -1.0
        self.best_score: float = 0.0
        self.selected_features: List[str] = []

    def _initialize_population(self, n_features: int) -> np.ndarray:
        """Initializes a diverse binary population."""
        # Biased random initialization (selecting ~30% - 60% of features initially)
        prob = self.rng.uniform(0.3, 0.6, size=(self.pop_size, 1))
        population = (self.rng.rand(self.pop_size, n_features) < prob).astype(int)
        
        # Ensure every individual has at least 1 feature
        for i in range(self.pop_size):
            if np.sum(population[i]) == 0:
                random_idx = self.rng.randint(0, n_features)
                population[i, random_idx] = 1
        return population

    def _evaluate_fitness(
        self,
        chromosome: np.ndarray,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates fitness:
        Fitness = Balanced_Accuracy - alpha * (k / D)
        """
        n_selected = int(np.sum(chromosome))
        total_features = len(chromosome)

        if n_selected == 0:
            return 0.0, 0.0

        selected_indices = np.where(chromosome == 1)[0]
        X_tr_sub = X_tr[:, selected_indices]
        X_val_sub = X_val[:, selected_indices]

        # Fast lightweight evaluator
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=self.random_state, solver="liblinear")
        try:
            clf.fit(X_tr_sub, y_tr)
            y_pred = clf.predict(X_val_sub)
            score = float(balanced_accuracy_score(y_val, y_pred))
        except Exception:
            score = 0.5  # Neutral fallback

        # Parsimony penalty for number of features
        penalty_val = self.penalty * (n_selected / total_features)
        fitness = max(0.0, score - penalty_val)
        return fitness, score

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        val_size: float = 0.25,
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None
    ) -> "GeneticFeatureSelector":
        """
        Executes the Genetic Algorithm optimization loop.
        """
        n_samples, n_features = X.shape

        # Internal train/val split strictly within the training fold
        X_tr, X_v, y_tr, y_v = train_test_split(
            X, y,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        population = self._initialize_population(n_features)
        self.history = []

        global_best_fitness = -1.0
        global_best_chromosome = None
        global_best_score = 0.0

        for gen in range(self.generations):
            fitness_scores = []
            raw_scores = []

            for individual in population:
                fit, raw = self._evaluate_fitness(individual, X_tr, y_tr, X_v, y_v)
                fitness_scores.append(fit)
                raw_scores.append(raw)

            fitness_arr = np.array(fitness_scores)
            raw_arr = np.array(raw_scores)

            best_idx = int(np.argmax(fitness_arr))
            gen_best_fitness = fitness_arr[best_idx]
            gen_best_score = raw_arr[best_idx]
            gen_avg_fitness = float(np.mean(fitness_arr))
            gen_best_k = int(np.sum(population[best_idx]))

            if gen_best_fitness > global_best_fitness:
                global_best_fitness = gen_best_fitness
                global_best_chromosome = population[best_idx].copy()
                global_best_score = gen_best_score

            self.history.append({
                "generation": gen + 1,
                "best_fitness": round(gen_best_fitness, 4),
                "avg_fitness": round(gen_avg_fitness, 4),
                "best_score": round(gen_best_score, 4),
                "selected_count": gen_best_k
            })

            if progress_callback:
                progress_callback(gen + 1, self.generations, gen_best_fitness, gen_best_k)

            # Generate next generation
            next_pop = []
            # Elitism: retain top 2 individuals
            elite_indices = np.argsort(fitness_arr)[-2:]
            for idx in elite_indices:
                next_pop.append(population[idx].copy())

            # Selection, Crossover, and Mutation for remaining slots
            while len(next_pop) < self.pop_size:
                # Tournament Selection
                p1 = self._tournament_select(population, fitness_arr)
                p2 = self._tournament_select(population, fitness_arr)

                # Uniform Crossover
                if self.rng.rand() < self.cx_rate:
                    mask = self.rng.rand(n_features) < 0.5
                    child = np.where(mask, p1, p2)
                else:
                    child = p1.copy()

                # Bit-flip Mutation
                mut_mask = self.rng.rand(n_features) < self.mut_rate
                child = np.bitwise_xor(child, mut_mask.astype(int))

                # Ensure at least 1 feature is selected
                if np.sum(child) == 0:
                    child[self.rng.randint(0, n_features)] = 1

                next_pop.append(child)

            population = np.array(next_pop)

        self.best_chromosome = global_best_chromosome
        self.best_fitness = global_best_fitness
        self.best_score = global_best_score
        
        selected_indices = np.where(self.best_chromosome == 1)[0]
        self.selected_features = [feature_names[i] for i in selected_indices]

        return self

    def _tournament_select(self, population: np.ndarray, fitness_arr: np.ndarray) -> np.ndarray:
        """Tournament selection operator."""
        candidates = self.rng.choice(len(population), size=min(self.tournament_size, len(population)), replace=False)
        best_cand = candidates[np.argmax(fitness_arr[candidates])]
        return population[best_cand].copy()


def run_genetic_feature_selection(
    X_train_processed: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    population_size: int = 15,
    generations: int = 8,
    mutation_rate: float = 0.05,
    feature_penalty: float = 0.05,
    max_eval_samples: int = 2500,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int, float, int], None]] = None
) -> Dict[str, Any]:
    """
    Wrapper function to execute reproducible GA feature selection on preprocessed training data.
    Subsamples rows to max_eval_samples if needed to ensure swift laptop execution.
    """
    n_samples = len(y_train)
    if n_samples > max_eval_samples:
        indices = np.random.RandomState(random_state).choice(n_samples, size=max_eval_samples, replace=False)
        X_sub = X_train_processed[indices]
        y_sub = np.asarray(y_train)[indices]
    else:
        X_sub = X_train_processed
        y_sub = np.asarray(y_train)

    selector = GeneticFeatureSelector(
        population_size=population_size,
        generations=generations,
        crossover_rate=0.8,
        mutation_rate=mutation_rate,
        feature_penalty=feature_penalty,
        random_state=random_state
    )

    selector.fit(
        X=X_sub,
        y=y_sub,
        feature_names=feature_names,
        progress_callback=progress_callback
    )

    history_df = pd.DataFrame(selector.history)

    return {
        "selected_features": selector.selected_features,
        "selected_indices": np.where(selector.best_chromosome == 1)[0].tolist(),
        "total_features": len(feature_names),
        "selected_count": len(selector.selected_features),
        "best_fitness": selector.best_fitness,
        "best_score": selector.best_score,
        "history": selector.history,
        "history_df": history_df,
        "chromosome": selector.best_chromosome.tolist()
    }


def plot_ga_convergence(ga_results: Dict[str, Any]) -> go.Figure:
    """Plots Best Fitness and Average Fitness across generations."""
    history = ga_results["history_df"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["generation"],
        y=history["best_fitness"],
        mode="lines+markers",
        name="Best Chromosome Fitness",
        line=dict(color=SOC_COLORS["accent"], width=3),
        marker=dict(size=8, color=SOC_COLORS["accent"])
    ))

    fig.add_trace(go.Scatter(
        x=history["generation"],
        y=history["avg_fitness"],
        mode="lines+markers",
        name="Population Average Fitness",
        line=dict(color=SOC_COLORS["secondary"], width=2, dash="dot"),
        marker=dict(size=6)
    ))

    fig.update_layout(
        **SOC_PLOT_LAYOUT,
        title="Genetic Algorithm: Fitness Convergence Curve",
        xaxis_title="Generation",
        yaxis_title="Fitness Score (Accuracy - Parsimony Penalty)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
