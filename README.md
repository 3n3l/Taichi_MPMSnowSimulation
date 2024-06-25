# Simulating snow with the material point method (MPM)
- as described in ['A material point method for snow simulation'](https://dl.acm.org/doi/pdf/10.1145/2461912.2461948)
- implemented with the ['Moving Least Squares Material Point Method, MLS-MPM'](https://dl.acm.org/doi/pdf/10.1145/3197517.3201293)
- written in [Taichi](https://docs.taichi-lang.org/) based on [their example code](https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples)

![snowballHitsWallSticky](https://github.com/3n3l/Taichi_MPMSnowSimulation/assets/103253630/890473f7-5e3d-4209-aab6-4d4454c1bed6)

![snowballHitsSnowball2](https://github.com/3n3l/Taichi_MPMSnowSimulation/assets/103253630/a3e70218-6251-48ce-ada6-531b5e5ed331)

## Dependencies
```
conda env create --file=environment.yaml
conda activate MPMSnowSimulation
```

## Usage
```
python src/main.py
```
