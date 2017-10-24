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
**Return of an action:**
The Return value of an action is an evaluation of the performance of that action on a certain period. Let $a$ be an action, and $V_i$ be the Value of $a$. The return value $R$ of $a$ can be computed as follows: 
$$ R (a, t_i) = \frac{V_i - V_{i - 1} + Cash \space flow \space received}{V_{i - 1}} $$
**Return of a portfolio:**
The Return value of a portfolio is an evaluation of the performance of that portfolio on a certain period. It is actually the weighted average of its actions. Let $P$ a portfolio of size $N$. Let $(a_i)_{i \in [[1, N]]}$ the list of actions in $P$ and their respective weights $(w_i)_{i \in [[1, N]]}$. The return value $R$ of $P$ can be computed as follows:
$$ R(P, t) = \sum_{i = 1}^N{w_i R(a_i, t)} $$

#### Volatility -- Volatilité (Fr)
**Volatility of an action:**
volatility $\sigma$ is the degree of variation of a trading price series over time as measured by the standard deviation of logarithmic returns _(Wikipedia)_. In _JUMP_'s slides, it is the standard deviation of the value. We will follow this definition since we are using their API. Let $a$ an action, $\sigma_a$ its volatility and $V_a$ its Variance on a period of size $n$. Let $\overline a$ the mean value of $a$ on that period and $a_i$ the value of $a$ at an instant $i$. The Volatility of $a$ can be computed as follows:
$$ \sigma_a = \sqrt{V_a} \\
\sigma_a = \sqrt{\frac{\sum_{i = 1}^n (a_i-\overline a)^2}{n}}
$$
**Volatility of a portfolio:**
The volatility of a portfolio is standard deviation of the value (or return) of that portfolio. Since any portfolio is a weighted sum of actions and we can compute the variance-covariance matrix of all actions, it is easy to compute the volatility of a portfolio. 
Let $P$ a porfolio of size $N$ sush as :

* $(a_i)_{i \in [[1, N]]}$ is the list actions composing $P$
* $(w_i)_{i \in [[1, N]]}$ is the list of respective weights
* The values of the actions are taken over a period $T$ of size $n$
* $a_{i,j}$ is the value of action $a_i$ at the instant $j$ in $T$
* $\overline a_i$ is the mean value of the action $a_i$ over $T$
* $Cov(a_i, a_j)$ the covariance between actions $a_i$ and $a_j$ over $T$

The mean and covariance over $T$ are computed as follows:

* $\overline a_i = \frac{1}{n}\sum_{j = 1}^n a_{i, j}$
* $Cov(a_i, a_j) = \frac{1}{n}\sum_{k = 1}^n (a_{i,k} - \overline a_i)(a_{j, k} - \overline a_j)$

Let $p$ the value of $P$, we have $p = \sum_{i=1}^N w_i a_i$. The volatility $\sigma_P$ of the portfolio is simply the standard deviation of $p$, there for $\sigma_P = \sqrt{Var(p)}$ with $Var(p)$ the variance of $p$ over the period $T$.

$$ \sigma_P = \sqrt{\sum_{i = 1}^N \sum_{j = 1}^N w_i w_j Cov(a_i, a_j)} $$

### Conclusion
In order to optimize a portfolio $P$ of size $N$ (already specified actions), we need to find the weights $(w_i)_{i \in [[1, N]]}$ that maximizes $R(P)$ and minimizes $\sigma_P$
