from ssspy.bss.cacgmm import CACGMM as CACGMMBase
from tqdm import tqdm
import numpy as np

class CACGMM(CACGMMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.progress_bar = None

    def __call__(
        self, input: np.ndarray, n_iter: int = 100, initial_call: bool = True, **kwargs
    ) -> np.ndarray:
        self.n_iter = n_iter

        return super().__call__(input, n_iter=n_iter, initial_call=initial_call, **kwargs)

    def update_once(self) -> None:
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=self.n_iter)

        super().update_once()
        self.progress_bar.update(1)