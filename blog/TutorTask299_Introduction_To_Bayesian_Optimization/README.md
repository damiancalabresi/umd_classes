# Introduction to Bayesian Optimization

This article provides a brief introduction to Bayesian Optimization. I'll go over the basics of this optimization technique and explain why you should use it.

## When should you use Bayesian Optimization?

Bayesian Optimization is ideal for black-box optimization problems.
Black-box optimization problems are those where the objective function is not known or is very complex, it cannot be expressed mathematically and usually requires to run a real-life experiment.

A good example of this is the investigation done at Meta Research to find a formula for *concrete* with optimal compressive strength and low CO2 emissions. In this case, the objective function (The *compressive strength*) cannot be expressed as a mathematical formula or predicted and must be obtained by running real-life experiments.

- [Engineering At Meta: Using AI to make lower-carbon, faster-curing concrete](https://engineering.fb.com/2025/07/16/data-center-engineering/ai-make-lower-carbon-faster-curing-concrete/)
- [Github: Sustainable Concrete](https://github.com/facebookresearch/SustainableConcrete)

The *core idea* of Bayesian Optimization is to leverage Bayesian statistics to predict which value has the highest probability of being the optimal one. The algorithm will then return those values to be tested experimentally. The result will then be reintroduced to the algorithm to adjust the probabilistic model and predict the next best value to test.

## Bayesian Optimization vs Global Search Methods

When maximizing a non-differentiable or a non-convex objective function, the "state-of-the-art" approach is to use global search methods. These methods are based on the idea of exploring large portions of the entire search space, wasting a lot of time and resources on non-promising parameters or values.

Most common global search methods are:

- **Genetic Algorithms**: Based on the idea of evolution and natural selection. Generate a population of solutions and selects the best ones to be used in the next generation. The next generation will have random modifications of those solutions to be evaluated.
- **Simulated Annealing**: Selects a random solution to evaluate. On each iteration, makes a random jump to a new solution to evaluate. If the new one is better, it will be kept. On each iteration, the magnitude of jump is reduced to focus on the best solutions.
- **Particle Swarm Optimization**: Starts with a population of particles (Solutions). The best global solution among all the particles is stored. On each iteration, the particles move towards the best global solution to find an optimal value.

### Exploration vs Exploitation

Exploration is the process of exploring the search space to find new solutions. Exploitation is the process of refining the search around the best solutions found to date.

Global Search methods focus mostly on exploration, using random jumps. Bayesian Optimization uses probabilities to define when a given solution should be explored or exploited.

## How does Bayesian Optimization work?

In a Bayesian Optimization problem we assume we're trying to optimize a black-box function $f(x)$, which contains some uncertainty given the nature of the problem. We represent this function with a **Surrogate Model** $y = f(x) + \epsilon$.

![Surrogate Model](images/surrogate-model.png)
*Image source: [Adaptive Experimentation (Ax) - ax.dev](https://ax.dev/)*

The function is defined as a **Gaussian Process (GP)** $f(x) \sim GP(m(x), k(x, x'))$ where:
- $m(x)$ is the mean function
- $k(x, x')$ is the covariance function (Usually a Gaussian kernel $k(x, x') = \sigma^2 \exp(-\frac{||x - x'||^2}{2l^2})$)

In other words:
- A mean is defined at each evaluated point $x_i$
- An uncertainty is defined in all the other points, defined by the covariance function
- The covariance function allows interpolation of unknown $f(x_i)$, with some uncertainty

### Acquisition Function

The Acquisition Function is the function used to select the next point to evaluate. The most commonly used is the **Expected Improvement (EI):** $EI(x)=E[\max(f(x)−f^*, 0)]$.

### Sequential Process

Given a series of points and the corresponding outcomes, the Bayesian model will define a Gaussian Process (GP). The Acquisition Function is then applied to the GP, to calculate which point, based on the uncertainty and the mean, has the highest probability of being the optimal one. This point is returned by the algorithm to be evaluated and continue adjusting the Gaussian Process.

The following animation shows how a single variable, basic Gaussian Process, evolves over time:

![Sequential Exploration](images/sequential-exploration.gif)
*Image source: [Adaptive Experimentation (Ax) - ax.dev](https://ax.dev/)*

## Adaptive Experimentation

Multiple libraries and frameworks are available to implement Bayesian Optimization. **Adaptive Experimentation (Ax)**, developed by Meta, is a high-level API that abstracts the users from the complexity of BoTorch. It provides a simple interface to configure and run an experiment.

The documentation is available at [Ax.dev](https://ax.dev/).

## A Bayesian Optimization Example - The Hartmann Function

To show the capabilities of this technique, I'm going to apply it to find the global optima of the Hartmann function in 6 dimensions.

The Hartmann Function is defined as:

$$
f(\mathbf{x}) = -\sum_{i=1}^{4} \alpha_i \exp \left( -\sum_{j=1}^{6} A_{ij} (x_j - P_{ij})^2 \right)
$$

Its definition makes this function have multiple local optima and one global optima.

The following is a visualization of a Hartmann Function in 2 dimensions:

<img src="images/hartmann-2d.png" alt="Hartmann Function" width="400">

Even when we know the formula, a global search method is required to find the global optima. This is how the previously mentioned methods behave in this case:

- Simulated Annealing: It will probably get stuck in a local optima.
- Genetic Algorithms: Requires too many experiments, the success rate would be totally random.
- Particle Swarm Optimization: Probably better, the local optima will move towards the global optima, but still requires too many experiments.

The domain of the function are 6 dimensions, each one between 0 and 1. Doing a grid search with a step of 0.1 would require 10^6 points to be evaluated. Using Bayesian Optimization, we were able to find the global optima in less than 45 experiments.

The code below shows how to use Ax to find the global optima.

```python
# Create the Ax Client
client = Client()

# The Hartmann function has 6 variables that go between 0 and 1.
parameters = [
    RangeParameterConfig( name="x1", parameter_type="float", bounds=(0, 1) ),
    RangeParameterConfig( name="x2", parameter_type="float", bounds=(0, 1) ),
    RangeParameterConfig( name="x3", parameter_type="float", bounds=(0, 1) ),
    RangeParameterConfig( name="x4", parameter_type="float", bounds=(0, 1) ),
    RangeParameterConfig( name="x5", parameter_type="float", bounds=(0, 1) ),
    RangeParameterConfig(name="x6", parameter_type="float", bounds=(0, 1)),
]

client.configure_experiment(parameters=parameters)

# The "-" sign transforms this into a minimization problem.
client.configure_optimization(objective="-hartmann")

# Define the optimization loop
for _ in range(10):
    # Request a series of inputs to Ax client
    trials = client.get_next_trials(max_trials=5)

    # Conduct the experiments (5 times in this case)
    for trial_index, parameters in trials.items():
        x1 = parameters["x1"]
        x2 = parameters["x2"]
        x3 = parameters["x3"]
        x4 = parameters["x4"]
        x5 = parameters["x5"]
        x6 = parameters["x6"]

        result = hartmann6(x1, x2, x3, x4, x5, x6)

        # Send the results back to Ax with the trial index for reference
        client.complete_trial(trial_index=trial_index, raw_data={"hartmann": result})

        best_parameters, prediction, index, name = client.get_best_parameterization()
        print("Best Parameters:", best_parameters)
        print("Prediction (mean, variance):", prediction)
```

The optimal value found by Ax was `-3.29` while we know the global optima for the Hartmann function is `-3.32`. Just a 0.9% difference with only 45 iterations.

## Next Steps

In the upcoming articles, I'll demonstrate how Bayesian Optimization can be leveraged to optimize real-world marketing campaigns, providing lower biding costs and higher revenue.

I'm also going to cover how to use Bayesian Optimization to resolve multi-objective optimization problems, finding a Pareto frontier that helps to trade-off between the two objectives.

## References

- [Engineering At Meta: Efficient Optimization With Ax, an Open Platform for Adaptive Experimentation](https://engineering.fb.com/2025/11/18/open-source/efficient-optimization-ax-open-platform-adaptive-experimentation/)
- [Engineering At Meta: Using AI to make lower-carbon, faster-curing concrete](https://engineering.fb.com/2025/07/16/data-center-engineering/ai-make-lower-carbon-faster-curing-concrete/)
- [Github: Sustainable Concrete](https://github.com/facebookresearch/SustainableConcrete)
- [Ax: A Platform for Adaptive Experimentation](https://openreview.net/forum?id=U1f6wHtG1g)