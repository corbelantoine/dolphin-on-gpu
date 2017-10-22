# Finance Recap
Here you can find a small recap on portfolios and metrics used to compare portfolios.

## Portfolio
A portfolio is a combination of actions with respective shares. In order to ease our process, we will represent shares as a percentage of the portfolio. We will represent a portfolio $P$ as follows :

* $P$ is of size $N$ if it has $N$ different action
* list of actions: $(a_i)_{i \in [[1, N]]}$ 
* list of weights: $(w_i)_{i \in [[1, N]]}$ 

To optimize a portfolio, we have to find the best combination of weights and actions. Suppose we have $M$ actions and we want to create the best portfolio of size $N$. It can be represented by the following function: 
$$F: (a_i)_{i \in [[1, M]]} \mapsto \begin{pmatrix} a_i \\ w_i \end{pmatrix}_{i \in [[1, N]]}$$
Then again, this function is supposed to give us the optimal list of actions ande their respective weights. We don't know yet what optimal means and how to compute it. Here comes the metrics.

### Metrics
In order to compare portfolios, we need to pass by some metrics. These metrics are defined for one action and can be generalized to a portfolio. In this section we will define the following metrics:

* Return (Rendement en français)
* Volatility (Volatilité en français)
* Sharpe 

#### Return -- Rendement (Fr)
The Return value of an action is an evaluation of the performance of that action on a certain period. Let $a$ be an action, and $V_i$ be the Value of $a$. The return value $R$ of $a$ can be computed as follows: 
$$ R (a, t_i) = \frac{V_i - V_{i - 1} + Cash \space flow \space received}{V_{i - 1}} $$
The Return value of a portfolio is an evaluation of the performance of that portfolio on a certain period. It is actually the weighted average of its actions. Let $P$ a portfolio of size $N$. Let $(a_i)_{i \in [[1, N]]}$ the list of actions in $P$ and their respective weights $(w_i)_{i \in [[1, N]]}$. The return value $R$ of $P$ can be computed as follows:
$$ R(P, t) = \sum_{i = 1}^N{w_i a_i} $$

#### Volatility -- Volatilité (Fr)
volatility $\sigma$ is the degree of variation of a trading price series over time as measured by the standard deviation of logarithmic returns.