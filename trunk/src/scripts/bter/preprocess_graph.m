function G = preprocess_graph(G)
%PREPROCESS_GRAPH Makes graph undirected, simple, and removes self links.
%
%   G = PROPROCESS_GRAPH(G) takes an adjancency matrix G, changes directed
%   edges to undirected, removes self links, and makes the graph simple
%   (i.e., all edge weights are 1).
%
%Distributed as part of the BTER Project, Sandia National Labs, 2011.
%For more information, contact Tamara G. Kolda, tgkolda@sandia.gov.

% Make undirected
if ~isequal(G,G')
    G = G + G';
end

% Remove self-links
G = spdiags(zeros(size(G,1),1),0,G);

% Make simple
G = spones(G); 
