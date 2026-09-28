import numpy as np
from poli_baselines.solvers.simple.random_mutation import RandomMutation
from poli.core.abstract_black_box import AbstractBlackBox



from pep_compass.optimization.black_box.negative_black_box import NegativeBlackBox
from pep_compass.optimization.optimizer import AbstractOptimizer
from pep_compass.utils.utils import set_seed


class RandomMutationOptimizer(AbstractOptimizer):
    
    def __init__(
        self,
        black_box: AbstractBlackBox,
    ):
        if not black_box.maximize:
            black_box = NegativeBlackBox(black_box)

        super().__init__(black_box)
    
    def optimize(
        self, 
        evaluation_budget: int, 
        starting_point, 
        rng_seed: int | None = None,
    ):
        set_seed(rng_seed)
        
        random_mutation_solver = RandomMutation(
            black_box=self.black_box,
            x0=np.array([starting_point]),
            y0=self.black_box(np.array([starting_point])),
        )
        
        solver_result = random_mutation_solver.solve(max_iter=evaluation_budget)
        
        result = {
            'best_x': "".join(solver_result[0][0]),
            'best_y': abs(solver_result[1].item()),
        }
        
        print(f"RandomMutationOptimizer result: {result}")
        return result