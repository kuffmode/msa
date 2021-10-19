# Multi-perturbation Shapley value Analysis (MSA)
**TLDR**: A Game theoretical approach for calculating the contribution of each element of a system (here network models of the brain) to a system-wide description of the system. The classic neuroscience example: How much each brain region is causally relevant to an arbitrary cognitive function.

## Motivation & such:
MSA is developed by Keinan and colleagues back in 2004, the motivation for them was to have a causal picture of the system by lesioning its elements. The method itself is not new, if not the first, it was among one of the earliest ones used by neuroscientists to understand the brain. The reasoning is quite simple, let us study broken systems to see what's missing both from the brain and the behavior (or cognition) and assume that region was causally necessary for the emergence of that cognitive/behavioral state. What MSA does is to see this necessity as contribution. If the brain region is indeed the seed of this cognitive function (whatever this means) then its contribution should be very high while other regions will have near zero contribution. Having this in mind then we can see the whole scenario as a cooperative game in which a coalition of players work together and obtain some divisible outcome, then the question is quite the same. How to divide the outcome to the players in a "fair" way such that the most "important" player gets the biggest chunk. Shapley value is then that chunk! It is the result of a mathematically rigorous and axiomatic procedure that derives who should get how much from all possible combinations of coalitions and all ordering in which players can enter the game. Translating it to neuroscience, it derives a ranking of contributions from a dataset of all possible combinations of lesions. This means 2<sup>N</sup> lesions (assuming lesions are binary, either perturbed or not), which N is the number of brain regions. 

As you probably noticed this won't be feasible to calclulate as for example, it means a total number of 4,503,599,627,370,496 lesion combinations, assuming the brain is organized as Broadmann said, i.e., with 52 regions. So we estimate! For a more detailed description visit:

[Keinan, Alon, Claus C. Hilgetag, Isaac Meilijson, and Eytan Ruppin. 2004. “Causal Localization of Neural Function: The Shapley Value Method.” Neurocomputing 58-60 (June): 215–22.
](https://www.sciencedirect.com/science/article/abs/pii/S0925231204000426?via%3Dihub)

And

[Keinan, Alon, Ben Sandbank, Claus C. Hilgetag, Isaac Meilijson, and Eytan Ruppin. 2006. “Axiomatic Scalable Neurocontroller Analysis via the Shapley Value.” Artificial Life 12 (3): 333–52.](https://direct.mit.edu/artl/article/12/3/333/2530/Axiomatic-Scalable-Neurocontroller-Analysis-via)

## How-to:
I will update this part with some easy-to-digest figures.

## TODO:
Update How-to.
