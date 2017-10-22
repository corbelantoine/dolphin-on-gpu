# Finance Recap
Here you can find a small recap on portfolios and metrics used to compare portfolios.

## Portfolio
A portfolio is a combination of actions with respective shares. In order to ease our process, we will represent shares as a percentage of the portfolio. We will represent a portfolio $P$ as follows :

* $P$ is of size $N$ if it has $N$ different action
* list of actions: $(a_i)_{i \in [[1, N]]}$ 
* list of weights: $(w_i)_{i \in [[1, N]]}$ 

To optimize a portfolio, we have to find the best combination of weights and actions. Suppose we have $M$ actions and we want to create the best portfolio of size $N$. It can be represented by the following function: 
$$F: (a_i)_{i \in [[1, M]]} \mapsto ((a_i,w_i)_{i \in [[1, N]]})$$
Then again, this function is supposed to give us the optimal list of actions ande their respective weights. We don't know yet what optimal means and how to compute it. Here comes the metrics.

### Metrics
