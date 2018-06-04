Neil Kumar (nk2739)
Data Challenge 1 (Part 2)
4/9/18

Description of Algorithm:
I first populate my "data_challenge_data.csv" file with information from the nodes given to us. In this csv file, I record a node's ID, label, degree, clustering coefficient, and two heuristic values, its own value and its total neighbors' value. A node's value is determined by its degree * (1 + its clustering coef.), and its total neighbor value is each of its neighbor's value weighted by its label (1 for A and 0.3 for B) summed together. When I write to this file, a node's neighbor value will be 0 unless that node's information has been requested either randomly or specifically. 

After visiting the initial seed set, I do some random calls to explore the graph and hopefully find nodes with high values. I then do a series of specific calls to the next biggest node that has not been explored (neighbor value of 0) and try to explore the graph even further since high value nodes typically have a lot of neighbors and could reveal some strong nodes. I alternate between visting random and specific('strongest') nodes until I exhaust all my calls. 

Finally, when choosing the top 250 nodes I sort all nodes I have seen based on a sum of their value and their neighbor value. Since all nodes that have not been explored have a neighbor value of 0, this algorithm tends to lean towards nodes that have been explored. But this is intentional because I purposefully explored nodes that had strong values themselves. In the end, I found 250 nodes that all had fairly high degrees, high clustering coefficients, and varied neighbors, so I think my algorithm will do well. 

Why I Chose This Algorithm:
Since our seed set will promote nodes regardless of their label, I wanted to value a node based on its degree and clustering coefficient only. Since more connected nodes will probably promote more, I multiplied every node's degree by (1+ clustering coef.), so there was no penalty for a coefficient of 0 but a doubling of value for a coeffient of 1. However, I did take into consideration a node's label when calculating a node's "Neighbor Value", which is the sum of all of a node's neighbor's "Value"(s) weighted by their labels (1 for A and 0.3 for B). This way, if a node has a lot of neighbors but they are all labeled 'B', it might not be as potentially influential as a node with fewer neighbors but all labeled 'A'. 

I then decided to rank nodes based on the total of their value and their "Neighbor Value", which is 0 for unexplored nodes. I did this because if I chose nodes based on their "Value" only, I would probably choose the nodes with just the highest degree. However, once I incorporate the "Neighbor Value" I can discriminate between high-degree nodes that have similar degrees but much different "Neighbor Value"(s) based on their specific neighbors. Therefore, I found that a mix of the two heuristics provided good evaluations and hopefully a strong seed set.  

How to Read / Run My Code:
I provided necessary comments to hopefully understand my code as clearly as possible. 

To test the code, you can run "python data_challenge_1_pt2.py", which will run the main function.

In the main function, I initially get the API key / number of calls left to know where I am. 

I then ran and wrote the given seed set nodes into the data csv file. 

I then alternated between exploring 'random' nodes and 'specific' (or most valuable nodes I had seen) until I exhausted all 500 calls.

Finally, I looked at all nodes I had explored, including their neighbors, and chose / printed the top 250 into a file called "seedset.txt". 

The file "data_challenge_data.csv" contains the results of every time I made a call to the 'api/nodes' endpoint, writing the current node and its neighbors' information. 