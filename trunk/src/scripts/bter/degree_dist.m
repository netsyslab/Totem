function ddist = degree_dist(G)
%DEGREE_DIST Compute the degree distribution of a graph
%
%   D = DEGREE_DIST(G) computes the degree distribution of G so that D(i)
%   is the number of nodes that have degree i. Isolated nodes are ignored. 
%
%Distributed as part of the BTER Project, Sandia National Labs, 2011.
%For more information, contact Tamara G. Kolda, tgkolda@sandia.gov.

dlist = full(sum(G,2));
tf = (dlist > 0);
dlist = dlist(tf);
ddist = accumarray(dlist,1);
